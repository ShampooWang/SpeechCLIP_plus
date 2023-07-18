import json
import logging
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F

from ..base import OrderedNamespace
from ..module import (
    AttentionDiversityLoss,
    ClipModel,
    Custom_WavLM,
    FairseqSpeechEncoder_Hubert,
    KeywordDiversityLoss,
    MLPLayers,
    S3prlSpeechEncoderPlus,
    SimpleCache,
    SupConLoss,
    losses,
    mutualRetrieval,
)
from ..optim import get_scheduler
from ..util import (
    compute_dynamic_keyword_neighbors,
    compute_fixed_keyword_neighbors,
    random_crop_max_length,
)
from ..util.embedding_visualization import draw_embedding_space_PCA
from ..util.metric import cosine_semantics
from .base_model import BaseLightningModel
from .model_branches import (
    CascadedBranch,
    CascadedBranch_dynamic,
    HybridBranch,
    HybridBranch_dynamic,
    ParallelBranch,
)

METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}

logger = logging.getLogger(__name__)


class GeneralBase(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            raise DeprecationWarning("Please use s3prl_plus")
        elif self.audio_encoder_type == "s3prl_plus":
            self.audio_encoder = S3prlSpeechEncoderPlus(**config.audio_encoder)
        elif self.audio_encoder_type == "FairseqHubert":
            self.audio_encoder = FairseqSpeechEncoder_Hubert(**config.audio_encoder)
        elif self.audio_encoder_type == "custom_wavlm":
            self.audio_encoder = Custom_WavLM(**config.audio_encoder)
        else:
            logger.warning("No audio encoder loaded")

        self.clip = ClipModel(
            **config.clip,
        )

        if hasattr(self, "audio_encoder"):
            self.audio_embd_dim = self.audio_encoder.out_dim

        self.subword_embd_dim = self.clip.model.token_embedding.weight.size(-1)

        self.recall_at = config.retrieval.recall_at

        self.criterion = getattr(losses, config.cl_loss.type)(**config.cl_loss.args)

        self.log_detokenize_results = config.log_setting.get(
            "log_detokenize_results", True
        )

        if self.log_detokenize_results:
            self.log_detokenize_results_every_n_epoch = config.log_setting.get(
                "log_detokenize_results_every_n_epoch", 5
            )

        self.keyword_num = None

    def forward_audio(
        self,
        wav,
        wav_len,
        return_hidden_states: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        if self.audio_encoder_type in ["s3prl_plus", "FairseqHubert", "custom_wavlm"]:
            return self.audio_encoder(
                wav, wav_len, return_hidden_states=return_hidden_states
            )
        else:
            raise NotImplementedError("Unknown type:{}".format(self.audio_encoder_type))

    def forward(self, batch, cal_loss: bool = True):
        raise NotImplementedError()

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss
        """
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        losses, log_metrics = self.forward(batch)[:2]

        return {"loss_feats": losses, "log_metrics": log_metrics}

    def training_step_end(self, outputs):
        if isinstance(outputs, dict):
            if "loss" in outputs:
                # training_step has already calculated the loss
                return torch.mean(outputs["loss"])
            elif "loss_feats" in outputs and "log_metrics" in outputs:
                losses = self.compute_loss(outputs["loss_feats"])
                log_metrics = outputs["log_metrics"]
                result = {
                    **{f"train_{k}": losses[k] for k in losses},
                    **{
                        f"train_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                            log_metrics[k]
                        )
                        for k in log_metrics
                    },
                }
                self.log_dict(
                    result,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
                return {"loss": losses["loss"]}
            else:
                print("outputs", outputs)
                raise NotImplementedError()
        else:
            print("outputs", outputs)
            raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        losses, log_metrics, others = self.forward(batch)

        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others["image_feat"] if "image_feat" in others else None
        text_feat = others["text_feat"] if "text_feat" in others else None
        id = others["id"]

        return_dict = {
            "id": id,
            "audio_feat": audio_feat,
        }
        if image_feat is not None:
            return_dict["image_feat"] = image_feat
        if text_feat is not None:
            return_dict["text_feat"] = text_feat

        if "vq_keywords" in others and others["vq_keywords"] is not None:
            vq_keywords = others["vq_keywords"]
            return_dict["vq_keywords"] = vq_keywords
            return_dict["gold_text"] = batch["text"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}

    def validation_step_end(self, outputs):
        assert isinstance(outputs, dict)
        losses = self.compute_loss(outputs["loss_feats"])

        log_metrics = outputs["log_metrics"]
        result = {
            **{f"val_{k}": losses[k] for k in losses},
            **{
                f"val_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                    log_metrics[k]
                )
                for k in log_metrics
            },
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k in outputs["others"]:
            if isinstance(outputs["others"][k], torch.Tensor):
                outputs["others"][k] = outputs["others"][k].detach().cpu()
        return outputs["others"]

    def validation_epoch_end(self, outputs, log_cos_semantics=False):
        RootDir = self.config.trainer.default_root_dir
        if "vq_keywords" in outputs[0].keys():
            if not os.path.exists(os.path.join(RootDir, "retokenizeText")):
                os.makedirs(os.path.join(RootDir, "retokenizeText"), exist_ok=True)
            if not os.path.exists(os.path.join(RootDir, "visualization")):
                os.makedirs(
                    os.path.join(RootDir, "visualization"),
                    exist_ok=True,
                )

            if (
                hasattr(self, "log_detokenize_results_every_n_epoch")
                and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
            ):
                ##############################
                ## calculate mean, variance ##
                ##############################
                all_keyword_embeddings = torch.cat(
                    [x["vq_keywords"] for x in outputs], dim=0
                )

                embeddings_stat_dict = {
                    "mean": {},
                    "std": {},
                    "norm": {},
                }

                if self.keyword_num is not None:
                    all_keyword_embeddings = all_keyword_embeddings.view(
                        all_keyword_embeddings.shape[0],
                        self.keyword_num,
                        all_keyword_embeddings.shape[-1],
                    )
                    kwNumRange = self.keyword_num
                else:
                    all_keyword_embeddings = all_keyword_embeddings.unsqueeze(1)
                    kwNumRange = 1

                for i in range(kwNumRange):
                    kw_index = "kw" if self.keyword_num is None else f"kw_{i}"
                    embeddings_stat_dict["mean"][kw_index] = torch.mean(
                        torch.mean(all_keyword_embeddings[:, i, :], dim=0)
                    )
                    embeddings_stat_dict["std"][kw_index] = torch.mean(
                        torch.std(all_keyword_embeddings[:, i, :], dim=0)
                    )
                    embeddings_stat_dict["norm"][kw_index] = torch.mean(
                        torch.norm(all_keyword_embeddings[:, i, :], p=2, dim=-1)
                    )
                if self.keyword_num is None:
                    all_keyword_embeddings.squeeze(1)

                tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()
                self.log(
                    "kw_mean_mse",
                    torch.norm(
                        torch.mean(
                            all_keyword_embeddings.view(-1, self.subword_embd_dim),
                            dim=0,
                        )
                        - torch.mean(tokenEmbeddings, dim=0),
                        p=2,
                    ),
                    sync_dist=True,
                )
                self.log(
                    "kw_std_mse",
                    torch.std(
                        torch.norm(
                            torch.std(
                                all_keyword_embeddings.view(-1, self.subword_embd_dim),
                                dim=0,
                            )
                            - torch.std(tokenEmbeddings, dim=0),
                            p=2,
                        )
                    ),
                )

                #################
                ## Drawing PCA ##
                #################
                self.log_draw_pca_every_n_epoch = getattr(
                    self.config.log_setting, "log_draw_pca_every_n_epoch", 0
                )
                if self.log_draw_pca_every_n_epoch > 0 and (
                    self.current_epoch % self.log_draw_pca_every_n_epoch == 0
                ):
                    draw_embedding_space_PCA(
                        kw_embs=all_keyword_embeddings,
                        gold_embs=tokenEmbeddings,
                        output_path=os.path.join(
                            RootDir,
                            "visualization/",
                            "pca_ep{}.pdf".format(self.current_epoch),
                        ),
                    )

                ###################################################################
                ## Collecting gold texts' tokens and outputting keyword features ##
                ###################################################################
                gold_texts = []
                keyword_embeddings_list = []
                feat_len_list = []
                for x in outputs:
                    for sent in x["gold_text"]:
                        gold_texts.append(
                            self.clip.tokenizer.decode(sent.squeeze().tolist())
                        )

                    if self.keyword_num is None:
                        feat_len_list += x["keywords_len"].tolist()
                        embdList = torch.split(
                            x["vq_keywords"],
                            (x["keywords_bsz"] * x["max_kw_num"]).tolist(),
                            dim=0,
                        )
                        keyword_embeddings_list.append(
                            [
                                embd.view(bsz, knum, -1)
                                for bsz, knum, embd in zip(
                                    x["keywords_bsz"], x["max_kw_num"], embdList
                                )
                            ]
                        )

                ####################################
                ## Retreiving keywords' neighbors ##
                ####################################
                retrieve_method = getattr(
                    self.config.model_settings.cascaded_branch.keyword,
                    "retrieve_method",
                    "cosine",
                )
                if retrieve_method not in ["cosine", "pseudo_inverse"]:
                    raise NotImplementedError(retrieve_method)
                if retrieve_method == "pseudo_inverse":
                    emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()
                else:
                    emb_pinv = None

                K = self.config.model_settings.cascaded_branch.keyword.get(
                    "detokenized_K_neighbors", 10
                )
                print("Detokenizing K={}".format((K)))
                TextDir = os.path.join(RootDir, "retokenizeText/")
                all_retok_outputs = []
                if self.keyword_num is not None:
                    kw_top_ret = [[] for _ in range(self.keyword_num)]
                    (
                        hit_rate_list,
                        kw_top_ret,
                        all_retok_outputs,
                    ) = compute_fixed_keyword_neighbors(
                        model=self,
                        K=K,
                        retreival_type=retrieve_method,
                        tokenEmbeddings=tokenEmbeddings,
                        all_keyword_embeddings=all_keyword_embeddings,
                        gold_texts=gold_texts,
                        emb_pinv=emb_pinv,
                    )
                    # print(kw_top_ret)
                    # with open(os.path.join(TextDir, f"kw_hit_ep{self.current_epoch}.json"), "w") as f_hit:
                    #     json.dump(kw_top_ret, f_hit, indent=4)

                    hit_rate = torch.FloatTensor(hit_rate_list) / len(gold_texts) * 100
                    print("kw_hit_rate", hit_rate)
                    self.log(
                        "kw_hit_rate",
                        {
                            "kw_{}".format(i): hit_rate[i].item()
                            for i in range(self.keyword_num)
                        },
                        sync_dist=True,
                    )
                else:
                    (
                        hit_rate_list,
                        all_retok_outputs,
                    ) = compute_dynamic_keyword_neighbors(
                        model=self,
                        K=K,
                        retreival_type=retrieve_method,
                        outputs=outputs,
                        tokenEmbeddings=tokenEmbeddings,
                        keyword_embeddings_list=keyword_embeddings_list,
                        gold_texts=gold_texts,
                        feat_len_list=feat_len_list,
                        emb_pinv=emb_pinv,
                    )

                    val_kw_hit_rate = sum(hit_rate_list) / len(hit_rate_list) * 100
                    print("val_kw_hit_rate", val_kw_hit_rate)
                    self.log(
                        "val_kw_hit_rate",
                        val_kw_hit_rate,
                        sync_dist=True,
                    )

                with open(
                    os.path.join(TextDir, f"keywords_ep{self.current_epoch}.json"), "w"
                ) as f_kw:
                    json.dump(all_retok_outputs, f_kw, indent=4)

                if log_cos_semantics:
                    cos_semantic_list = cosine_semantics(all_retok_outputs)
                    val_cos_semantics = sum(cos_semantic_list) / len(cos_semantic_list)
                    print("val_cos_semantics", val_cos_semantics)
                    self.log(
                        "val_cos_semantics",
                        val_cos_semantics,
                        sync_dist=True,
                    )

                del all_retok_outputs

        ##################################
        ## Retreiving images and audios ##
        ##################################
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_imgs = torch.cat([x["image_feat"] for x in outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))

        torch.save(
            all_audo_feats.detach().cpu(),
            os.path.join(RootDir, "all_audio_feats.pt"),
        )
        torch.save(
            all_img_feats.detach().cpu(),
            os.path.join(RootDir, "all_img_feats.pt"),
        )

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats), len(all_audo_feats)
            )
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float().to(self.device),
            all_img_feats.float().T.to(self.device),
        )
        score_per_image = score_per_audio.T

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_image,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        image_feat = self.clip.encode_image(image_tensor)
        return image_feat

    def forward_text(self, sents: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(sents, list):
            textTensor = self.clip.prep_text(sents).to(self.device)
        elif isinstance(sents, torch.Tensor):
            if sents.dim() != 2:
                raise ValueError(f"Incorrect text tensor shape {sents.shape}")
            textTensor = sents
        else:
            raise TypeError(f"Unknown text type {type(sents)}")

        if hasattr(self.clip, "original2Reduced"):
            for i in range(textTensor.shape[0]):
                for j in range(textTensor.shape[1]):
                    textTensor[i, j] = self.clip.original2Reduced[
                        textTensor[i, j].item()
                    ]

        text_feat = self.clip.encode_text(textTensor)
        return text_feat

    def reportRetrieval(self, score_per_audio, score_per_image, AI_answers, IA_answers):
        recall_results_AI, recall_results_IA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
            recall_at=self.recall_at,
        )

        print("recall_results_AI", recall_results_AI)
        print("val_recall_IA", recall_results_IA)
        print("val_recall_mean", recall_results_mean)

        if isinstance(self.logger, WandbLogger):
            self.log("val_recall_AI", recall_results_AI, sync_dist=True)
            self.log("val_recall_IA", recall_results_IA, sync_dist=True)
            self.log("val_recall_mean", recall_results_mean, sync_dist=True)
        else:
            self.logger.experiment.add_scalars(
                "val_recall_AI", recall_results_AI, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_IA", recall_results_IA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_mean", recall_results_mean, self.global_step
            )
        self.log("val_recall_mean_10", recall_results_mean["recall@10"], sync_dist=True)

    def processWavs(self, wav):
        wav_len = [len(x) for x in wav]
        if isinstance(wav, torch.Tensor):
            wav_len = torch.LongTensor(wav_len, device=wav.device)
        return wav, wav_len

    def feature_extractor_s3prl(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ) -> torch.Tensor:
        """feature_extractor_s3prl
        Implement for s3prl to get feature
        Args:
            wav ():
        """
        raise NotImplementedError()

    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []

        if hasattr(self, "audio_encoder"):
            my_params += self.audio_encoder.trainable_params()
            my_params += list(self.criterion.parameters())

        my_params += self.clip.trainable_params()

        return my_params

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        my_params = self.getTrainableParams()

        audio_optimizer = getattr(torch.optim, self.config.audio_encoder.optim.name)(
            my_params,
            **self.config.audio_encoder.optim.args,
        )
        audio_scheduler = get_scheduler(
            optimizer=audio_optimizer,
            **self.config.audio_encoder.scheduler,
        )

        optimizers.append(audio_optimizer)
        schedulers.append(
            {
                "scheduler": audio_scheduler,
                "interval": "step",
            }
        )

        return optimizers, schedulers


class SpeechCLIP_plus(GeneralBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.cascaded_branch = None
        self.parallel_branch = None

        if self.config.model_settings.cascaded_objective_weight > 0:
            logger.info("Create branch of cascaded")
            if self.config.model_settings.cascaded_branch.type == "CascadedBranch":
                self.cascaded_branch = CascadedBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif (
                self.config.model_settings.cascaded_branch.type
                == "CascadedBranch_dynamic"
                or self.config.model_settings.cascaded_branch.type
                == "CascadedBranch_plus"
            ):
                self.cascaded_branch = CascadedBranch_dynamic(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif self.config.model_settings.cascaded_branch.type == "HybridBranch":
                assert (
                    self.config.model_settings.parallel_objective_weight > 0
                ), self.config.model_settings.parallel_objective_weight
                logger.info("Using Parallel Objective (Integrated w/ cascaded_branch)")
                self.cascaded_branch = HybridBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif (
                self.config.model_settings.cascaded_branch.type
                == "HybridBranch_dynamic"
            ):
                assert (
                    self.config.model_settings.parallel_objective_weight > 0
                ), self.config.model_settings.parallel_objective_weight
                logger.info(
                    "Using Parallel Objective (Integrated w/ cascaded_branch) and dynamic numbers of keywords"
                )
                self.cascaded_branch = HybridBranch_dynamic(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            else:
                raise NotImplementedError(
                    self.config.model_settings.cascaded_branch.type
                )

            if hasattr(self.cascaded_branch, "keyword_num"):
                self.keyword_num = self.cascaded_branch.keyword_num

            if self.config.model_settings.cascaded_branch.downsampling.type == "cif":
                self.quantity_loss_weight = getattr(
                    self.config.model_settings.cascaded_branch.downsampling.cif,
                    "quantity_loss_weight",
                    1.0,
                )
                self.quantity_loss_criteria = nn.L1Loss()

        if (
            self.config.model_settings.parallel_objective_weight > 0
            and self.cascaded_branch is None
        ):
            logger.info("Create Parallel Branch")
            self.parallel_branch = ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                text_dim=self.subword_embd_dim,
            )

        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )

        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )

        self.c_branch_proj_net = None
        cascaded_branch_projection = self.config.model_settings.get(
            "cascaded_branch_projection", None
        )
        if cascaded_branch_projection is not None:
            logger.info(
                f"cascaded_branch_projection dims:{cascaded_branch_projection.dimensions} droupout:{cascaded_branch_projection.dropout}"
            )
            self.c_branch_proj_net = MLPLayers(
                units=cascaded_branch_projection.dimensions,
                dropout=cascaded_branch_projection.dropout,
            )

        ############################
        ## Keyword diversity loss ##
        ############################
        self.keyword_diversity_weight = (
            config.model_settings.cascaded_branch.keyword.get("diversity_weight", 0.0)
        )
        if self.keyword_diversity_weight > 0:
            logger.info(
                f"Adding keyword diversity objective, weight: {self.keyword_diversity_weight}"
            )
            self.keyword_diversity_type = (
                config.model_settings.cascaded_branch.keyword.get(
                    "diversity_type", ["cos"]
                )
            )
            if isinstance(self.keyword_diversity_type, str):
                self.keyword_diversity_type = [self.keyword_diversity_type]
            assert isinstance(self.keyword_diversity_type, list), type(
                self.keyword_diversity_type
            )

            logger.info(f"Keyword diversity type: {self.keyword_diversity_type}")
            assert set(self.keyword_diversity_type).issubset(
                {"cos", "corr", "ent"}
            ), f"The implemented types of diversity loss are ['cos', 'corr', 'ent'], your target diversity type: {self.keyword_diversity_type}"

            if (
                "corr" in self.keyword_diversity_type
                or "cos" in self.keyword_diversity_type
            ):
                self.keyword_diversity_criterion = KeywordDiversityLoss(
                    self.keyword_diversity_weight
                )

        ##############################
        ## Attention diversity loss ##
        ##############################
        if self.cascaded_branch is None:
            self.attention_diversity_weight = 0.0
        else:
            if not hasattr(self.cascaded_branch, "self_att"):
                self.attention_diversity_weight = 0.0
            else:
                self.attention_diversity_weight = (
                    config.model_settings.cascaded_branch.transformer_args.get(
                        "attn_diversity_weight", 0.0
                    )
                )

        if self.attention_diversity_weight > 0:
            logger.info(
                f"Adding attention diversity loss, weight: {self.attention_diversity_weight}"
            )
            self.attention_diversity_criterion = AttentionDiversityLoss(
                self.attention_diversity_weight
            )

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        if self.cascaded_branch is not None:
            logger.info("Add cascaded_branch parameters")
            _params += list(
                filter(lambda x: x.requires_grad, self.cascaded_branch.parameters())
            )

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        if self.c_branch_proj_net is not None:
            logger.info("Add cascaded_branch_projection parameters")
            _params += list(self.c_branch_proj_net.parameters())

        return _params

    def forward(
        self,
        batch,
    ) -> dict:
        self.clip.update_device(self.device)  # update device information to clip model

        if self.training:
            batch = random_crop_max_length(
                batch, self.config.audio_encoder.max_audio_len
            )

        image, id, wav, wav_len = (
            batch["image"],
            batch["id"],
            batch["wav"],
            batch["wav_len"],
        )
        audio_feat, audio_feat_len = self.forward_audio(
            wav, wav_len, return_hidden_states=False
        )

        image_feat = self.forward_image(image)
        if self.img_enc_proj_net is not None:
            image_feat = self.img_enc_proj_net(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        parallel_audio_feat = (
            cascaded_audio_feat
        ) = vq_results = vq_keywords = dsample_results = keywords_len = None
        if self.cascaded_branch is not None:
            if isinstance(self.cascaded_branch, CascadedBranch_dynamic):
                otherInputs = {}
                if "fp_alignment" in batch:
                    otherInputs["fp_alignment"] = batch["fp_alignment"]
                if "segment_num" in batch:
                    otherInputs["dsample_target_len"] = batch["segment_num"]
                else:
                    if (
                        getattr(self.cascaded_branch, "using_gt_len", False)
                        and "text" in batch
                    ):
                        otherInputs["text"] = batch["text"]
                        dsample_target_len = torch.LongTensor(
                            [
                                (t.squeeze().tolist().index(49407) - 1)
                                for t in batch["text"]
                            ]
                        ).to(wav.device)
                    else:
                        otherInputs["text"] = None
                        dsample_target_len = (audio_feat_len / 20).round().long()
                    otherInputs["dsample_target_len"] = dsample_target_len

                if type(self.cascaded_branch) == HybridBranch_dynamic:
                    (
                        parallel_audio_feat,
                        cascaded_audio_feat,
                        vq_results,
                        vq_keywords,
                        dsample_results,
                    ) = self.cascaded_branch(
                        audio_feat=audio_feat,
                        audio_feat_len=audio_feat_len,
                        otherInputs=otherInputs,
                    )
                else:
                    (
                        cascaded_audio_feat,
                        vq_results,
                        vq_keywords,
                        dsample_results,
                    ) = self.cascaded_branch(
                        audio_feat=audio_feat,
                        audio_feat_len=audio_feat_len,
                        otherInputs=otherInputs,
                    )
                keywords_len = dsample_results["dsample_feats_length"]
            else:
                if type(self.cascaded_branch) == HybridBranch:
                    (
                        parallel_audio_feat,
                        cascaded_audio_feat,
                        vq_results,
                        vq_keywords,
                    ) = self.cascaded_branch(
                        audio_feat=audio_feat,
                        audio_feat_len=audio_feat_len,
                    )
                else:
                    cascaded_audio_feat, vq_results, vq_keywords = self.cascaded_branch(
                        audio_feat=audio_feat,
                        audio_feat_len=audio_feat_len,
                    )

        if self.parallel_branch is not None:
            parallel_audio_feat = self.parallel_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )

        ####################
        ## Extract losses ##
        ####################
        losses = {
            "id": id,
            "image_feat": image_feat,
        }
        if cascaded_audio_feat is not None:
            if self.c_branch_proj_net is not None:
                cascaded_audio_feat = self.cascaded_branch(cascaded_audio_feat)
            cascaded_audio_feat = cascaded_audio_feat / cascaded_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["cascaded_audio_feat"] = cascaded_audio_feat

        if parallel_audio_feat is not None:
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

        # if "attention_maps" in branchResultDict and hasattr(
        #     self, "attention_diversity_criterion"
        # ):
        #     losses["attention_diversity_loss"] = self.attention_diversity_criterion(
        #         branchResultDict["attention_maps"]
        #     )

        if self.cascaded_branch is not None:
            if self.keyword_diversity_weight > 0:
                assert vq_keywords is not None
                for div_type in self.keyword_diversity_type:
                    if div_type == "ent":
                        assert (
                            "diversity_loss" in vq_results
                        ), "entropy loss is not in vq_results"
                        losses["ent_diversity_loss"] = vq_results["diversity_loss"]
                    else:
                        if self.keyword_num is not None:
                            keyword_diversity_loss = self.keyword_diversity_criterion(
                                vq_keywords,
                                [self.keyword_num] * vq_keywords.shape[0],
                                div_type,
                            )
                        else:
                            keyword_diversity_loss = self.keyword_diversity_criterion(
                                vq_keywords, keywords_len, div_type
                            )
                        losses[f"{div_type}_diversity_loss"] = keyword_diversity_loss

            if (
                hasattr(self.cascaded_branch, "downsampling_type")
                and self.cascaded_branch.downsampling_type == "cif"
            ):
                assert (
                    "target_len" in dsample_results
                    and "quantity_out" in dsample_results
                ), f"{dsample_results.keys()}"
                losses["cif_quantity_out"] = dsample_results["quantity_out"]
                losses["cif_target_len"] = dsample_results["target_len"]

        #####################
        ## Logging metrics ##
        #####################
        log_metrics = {"cl_temp": self.criterion.current_temperature}

        if vq_results is not None:
            log_metrics["softmax_temp"] = vq_results["temp"]

        if self.cascaded_branch is not None:
            if dsample_results is not None and "dsample_len_diff" in dsample_results:
                log_metrics["dsample_len_diff"] = dsample_results["dsample_len_diff"]
            vq_results_log_keys = [
                "temp",
                "code_perplexity",
                "prob_perplexity",
                "ent_per_t",
            ]
            assert set(vq_results_log_keys).issubset(
                set(vq_results.keys())
            ), f"log keys: {vq_results_log_keys}, result: {vq_results.keys()}"
            vq_log_dict = {k: vq_results[k] for k in vq_results_log_keys}
            log_metrics.update(vq_log_dict)

        return (
            losses,
            log_metrics,
            {
                "id": id,
                "image_feat": image_feat,
                "parallel_audio_feat": parallel_audio_feat,
                "cascaded_audio_feat": cascaded_audio_feat,
                "vq_results": vq_results,
                "vq_keywords": vq_keywords,
                "dsample_results": dsample_results,
                "keywords_len": keywords_len,
            },
        )

    def compute_loss(self, inputDict: dict):
        assert isinstance(inputDict, dict)
        required_keys = {"id", "image_feat"}
        assert required_keys.issubset(
            set(inputDict.keys())
        ), f"required: {required_keys}, input: {inputDict.keys()}"

        losses = {"loss": 0}
        image_feat = inputDict["image_feat"].float()
        id = inputDict["id"]

        branchTypeList = ["cascaded", "parallel"]
        for branchType in branchTypeList:
            loss_weight = getattr(
                self.config.model_settings, f"{branchType}_objective_weight", 0.0
            )
            if loss_weight > 0.0:
                feats_key = f"{branchType}_audio_feat"
                assert feats_key in inputDict, f"{inputDict.keys()}"
                losses[f"{branchType[0]}_cl_loss"] = self.criterion(
                    feat_A=inputDict[feats_key].float(),
                    feat_B=image_feat,
                    index=id,
                )
                losses["loss"] += loss_weight * losses[f"{branchType[0]}_cl_loss"]

        for k, v in inputDict.items():
            if "diversity_loss" in k:
                losses[k] = METRIC_REDUCEFN_MAPPING[type(v)](v)
                losses["loss"] += self.keyword_diversity_weight * losses[k]

        if "attention_diversity_loss" in losses:
            losses["attention_diversity_loss"] = METRIC_REDUCEFN_MAPPING[
                type(inputDict["attention_diversity_loss"])
            ](inputDict["attention_diversity_loss"])
            losses["loss"] += (
                self.attention_diversity_weight * losses["attention_diversity_loss"]
            )

        if (
            "cif_quantity_out" in inputDict
            and "cif_target_len" in inputDict
            and hasattr(self, "quantity_loss_criteria")
        ):
            losses["quantity_loss"] = self.quantity_loss_criteria(
                inputDict["cif_quantity_out"], inputDict["cif_target_len"]
            )
            losses["loss"] += self.quantity_loss_weight * losses["quantity_loss"]

        return losses

    def validation_step(self, batch, batch_idx):
        losses, log_metrics, others = self.forward(batch)

        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others.get("image_feat", None)
        text_feat = others.get("text_feat", None)
        id = others["id"]

        returnDict = {
            "id": id,
            "audio_feat": audio_feat,
        }

        if image_feat is not None:
            returnDict["image_feat"] = image_feat
        if text_feat is not None:
            returnDict["text_feat"] = text_feat

        if "vq_keywords" in others and others["vq_keywords"] is not None:
            kwDict = {"gold_text": batch["text"]}
            vq_keywords = others["vq_keywords"]
            if self.keyword_num is not None:
                kwDict["vq_keywords"] = vq_keywords
            else:
                # Dynamic number of keywords
                kwDict.update(
                    {
                        "vq_keywords": vq_keywords.view(-1, vq_keywords.shape[-1]),
                        "keywords_bsz": vq_keywords.shape[0],
                        "max_kw_num": vq_keywords.shape[1],
                    }
                )

            returnDict.update(kwDict)

        if others["keywords_len"] is not None:
            returnDict["keywords_len"] = others["keywords_len"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": returnDict}

    def validation_step_end(self, outputs):
        assert isinstance(outputs, dict)
        losses = self.compute_loss(outputs["loss_feats"])

        log_metrics = outputs["log_metrics"]
        result = {
            **{f"val_{k}": losses[k] for k in losses},
            **{
                f"val_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                    log_metrics[k]
                )
                for k in log_metrics
            },
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k in outputs["others"]:
            if isinstance(outputs["others"][k], torch.Tensor):
                outputs["others"][k] = outputs["others"][k].detach().cpu()
        return outputs["others"]

    def feature_extractor_s3prl(self, wav):
        wav_len = [len(x) for x in wav]
        batch = {"wav": wav, "wav_len": wav_len}
        audio_feat, audio_feat_len, hidden_states = self.forward_audio(
            batch, return_hidden_states=True
        )
        hidden_states = [x for x in hidden_states]
        if self.parallel_branch is not None:
            additional_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_feat_len
            )
        else:
            if hasattr(self, "self_att"):
                additional_hidden_states = self.cascaded_branch.extract_hidden_states(
                    audio_feat, audio_feat_len
                )
            else:
                additional_hidden_states = []

        hidden_states = hidden_states + additional_hidden_states

        return hidden_states

    def feature_extractor_zerospeech(self, wav, feat_select_idx):
        feat_select_idx = int(feat_select_idx)
        result = []
        wav_len = [len(x) for x in wav]
        batch = {"wav": wav, "wav_len": wav_len}
        audio_feat, audio_feat_len, hidden_states = self.forward_audio(
            batch, return_hidden_states=True
        )
        hidden_states = [x for x in hidden_states]
        if self.parallel_branch is not None:
            additional_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_feat_len
            )
        else:
            if hasattr(self, "self_att"):
                additional_hidden_states = self.cascaded_branch.extract_hidden_states(
                    audio_feat, audio_feat_len
                )
            else:
                additional_hidden_states = []

        hidden_states = hidden_states + additional_hidden_states
        embeddings = hidden_states[feat_select_idx]

        for _embs, _len in zip(embeddings, audio_feat_len):
            result.append(_embs[:_len].cpu().float().numpy())

        return result

    def extract_keyword_boundary(self, wav, alignment):
        fp_alignment = []
        for ali in alignment:
            fp_alignment.extend([round(ali[0] * 50), round(ali[1] * 50)])
        segment_num = torch.LongTensor([len(fp_alignment)]).to(wav.device)

        fp_alignment = torch.LongTensor(fp_alignment).unique().to(wav.device)
        batch = {"wav": wav, "wav_len": [len(wav)], "fp_alignment": fp_alignment[-1]}
        embeddings, feat_len = self.forward_audio(batch)

        inputDict = {
            "audio_feat": embeddings,
            "audio_feat_len": feat_len,
            "fp_alignment": fp_alignment.unsqueeze(0),
            "dsample_target_len": segment_num,
        }
        result = self.cascaded_branch(inputDict)
        result["vq_results"]["targets"] = result["vq_results"]["targets"].view(
            -1,
        )
        result["vq_results"]["targets"] = [
            self.clip.reducedl2Original[_id.item()]
            for _id in result["vq_results"]["targets"]
        ]
        return result

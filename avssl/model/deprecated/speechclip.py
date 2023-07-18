class SpeechCLIP(GeneralBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        self.cascaded_branch = None
        self.parallel_branch = None
        if self.config.model_settings.cascaded_objective_weight > 0:
            logger.info("Create Cascaded Branch")
            if self.config.model_settings.cascaded_branch.type == "CascadedBranch":
                self.cascaded_branch = CascadedBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif self.config.model_settings.cascaded_branch.type == "HybridBranch":
                assert self.config.model_settings.parallel_objective_weight > 0
                logger.info("Using Parallel Objective (Integrated w/ cascaded_branch)")
                self.cascaded_branch = HybridBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            else:
                raise NotImplementedError()
            self.keyword_num = self.cascaded_branch.keyword_num
        if (
            self.config.model_settings.parallel_objective_weight > 0
            and not self.config.model_settings.cascaded_branch.type == "HybridBranch"
        ):
            logger.info("Create Parallel Branch")
            self.parallel_branch = ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
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

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        if self.cascaded_branch is not None:
            logger.info("Add cascaded_branch parameters")
            _params += list(self.cascaded_branch.parameters())

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        return _params

    def feature_extractor_s3prl(self, wav, featrure_layer_norm=True, add_feat=True):
        wav, wav_len = self.processWavs(wav)

        audio_feat, audio_len, hidden_states = self.forward_audio(
            wav, wav_len, return_hidden_states=True
        )
        hidden_states = [x for x in hidden_states]

        if add_feat:
            if self.cascaded_branch is not None:
                cascaded_hidden_states = self.cascaded_branch.extract_hidden_states(
                    audio_feat, audio_len
                )
                hidden_states = hidden_states + cascaded_hidden_states

            if self.parallel_branch is not None:
                parallel_hidden_states = self.parallel_branch.extract_hidden_states(
                    audio_feat, audio_len
                )
                hidden_states = hidden_states + parallel_hidden_states

        # assert featrure_layer_norm == True
        if featrure_layer_norm:
            hidden_states = torch.stack(hidden_states, dim=0)
            hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        hidden_states = [x for x in hidden_states]

        return hidden_states[-1], hidden_states

    def compute_loss(self, inputDict):
        assert isinstance(inputDict, dict)
        required_keys = {"id", "image_feat"}
        assert required_keys.issubset(
            set(inputDict.keys())
        ), f"required: {required_keys}, input: {inputDict.keys()}"

        losses = {"loss": 0}
        image_feat = inputDict["image_feat"].float()
        id = inputDict["id"]

        branchTypeList = ["cascaded", "parallel", "keywords"]
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

        return losses

    def forward(
        self,
        batch,
    ) -> dict:
        self.clip.update_device(self.device)  # update device information to clip model
        ############################
        ## Extract image features ##
        ############################
        image = batch["image"]
        image_feat = self.forward_image(image)
        if self.img_enc_proj_net is not None:
            image_feat = self.img_enc_proj_net(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        ##########################################
        ## Extract cascaded & parallel features ##
        ##########################################
        audio_feat, audio_feat_len, hidden_states = self.forward_audio(
            batch, return_hidden_states=True
        )

        if self.cascaded_branch is not None:
            resuldDict = self.cascaded_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )
            if self.c_branch_proj_net is not None:
                resuldDict["cascaded_audio_feat"] = self.c_branch_proj_net(
                    resuldDict["cascaded_audio_feat"]
                )

        if self.parallel_branch is not None:
            resuldDict = self.parallel_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )
            if self.p_branch_proj_net is not None:
                resuldDict["parallel_audio_feat"] = self.p_branch_proj_net(
                    resuldDict["parallel_audio_feat"]
                )

        resuldDict["id"] = batch["id"]
        resuldDict["image_feat"] = image_feat

        losses = {
            "id": batch["id"],
            "image_feat": image_feat,
        }
        if resuldDict["cascaded_audio_feat"] is not None:
            resuldDict["cascaded_audio_feat"] = resuldDict[
                "cascaded_audio_feat"
            ] / resuldDict["cascaded_audio_feat"].norm(dim=-1, keepdim=True)
            losses["cascaded_audio_feat"] = resuldDict["cascaded_audio_feat"]

        if resuldDict["parallel_audio_feat"] is not None:
            resuldDict["parallel_audio_feat"] = resuldDict[
                "parallel_audio_feat"
            ] / resuldDict["parallel_audio_feat"].norm(dim=-1, keepdim=True)
            losses["parallel_audio_feat"] = resuldDict["parallel_audio_feat"]

        log_metrics = {"cl_temp": self.criterion.current_temperature}
        if resuldDict["vq_results"] is not None:
            log_metrics["softmax_temp"] = resuldDict["vq_results"]["temp"]

        return losses, log_metrics, resuldDict

    def get_attention_weights(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ):
        wav_len = [len(x) for x in wav]
        self.clip.update_device(self.device)
        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        return self.cascaded_branch.getAttentionMap(audio_feat, audio_len)

    def feature_extractor_zerospeech(self, wav, feat_select_idx):
        feat_select_idx = int(feat_select_idx)
        result = []
        batch = {"wav": wav, "wav_len": [len(x) for x in wav]}
        audio_feat, audio_len, hidden_states = self.forward_audio(
            batch, return_hidden_states=True
        )
        hidden_states = [x for x in hidden_states]
        if self.cascaded_branch is not None:
            addtional_hidden_states = self.cascaded_branch.extract_hidden_states(
                audio_feat, audio_len
            )
        else:
            addtional_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_len
            )
        hidden_states = hidden_states + addtional_hidden_states
        embeddings = hidden_states[feat_select_idx]

        # if feat_select_idx < len_audio_encoder:
        #     for _embs, _len in zip(embeddings, audio_len):
        #         result.append(_embs[:_len].cpu().float().numpy())
        # elif feat_select_idx == len(hidden_states) - 1:
        #     keywords = embeddings["cif_out"]
        #     keywords_len= embeddings["cif_outputs_len"]
        #     for _k in keywords:
        #         assert _k.dim() == 2
        #         if _k.shape[0] == 1:
        #             _k = _k.repeat(2, 1)
        #         result.append(_k.cpu().float().numpy())
        # else:
        for _embs, _len in zip(embeddings, audio_len):
            result.append(_embs[:_len].cpu().float().numpy())

        return result

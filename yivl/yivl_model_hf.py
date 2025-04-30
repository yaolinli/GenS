from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.models.llava import (
    LlavaForConditionalGeneration,
    LlavaPreTrainedModel,
)
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from yivl.resampler import Resampler
from yivl.siglip_navit_490 import (
    CustomSiglipVisionConfig,
    CustomSiglipVisionModel,
)

from .yivl_config import YiVLConfig

AutoConfig.register("custom_siglip_vision_model", CustomSiglipVisionConfig)
AutoModel.register(CustomSiglipVisionConfig, CustomSiglipVisionModel)


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: YiVLConfig):
        super().__init__()
        if config.mlp_post_norm:
            modules = [
                nn.Linear(
                    config.vision_config.hidden_size, config.text_config.hidden_size
                ),
                nn.LayerNorm(config.text_config.hidden_size),
            ]
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(
                    config.text_config.hidden_size, config.text_config.hidden_size
                )
            )
            modules.append(nn.LayerNorm(config.text_config.hidden_size))
            self.vit_norm = nn.Identity()
        else:
            modules = [
                nn.Linear(
                    config.vision_config.hidden_size, config.text_config.hidden_size
                )
            ]
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(
                    config.text_config.hidden_size, config.text_config.hidden_size
                )
            )
            self.vit_norm = nn.LayerNorm(
                config.vision_config.hidden_size,
                elementwise_affine=config.mlp_post_norm_element_affine,
            )
        self.layers = nn.Sequential(*modules)

    def forward(self, image_features):
        image_features = self.vit_norm(image_features)
        return self.layers(image_features)


def build_mm_projector(config: YiVLConfig):
    if config.mm_projector == "mlp":
        return LlavaMultiModalProjector(config)
    elif config.mm_projector == "resampler":
        return Resampler(
            num_queries=config.tokens_per_image,
            embed_dim=config.vision_config.hidden_size,
            num_heads=config.vision_config.num_attention_heads,
            kv_dim=config.vision_config.hidden_size,
            ff_dim=config.text_config.hidden_size,
            output_dim=config.text_config.hidden_size,
            num_patches=(
                config.vision_config.image_size // config.vision_config.patch_size
            )
            ** 2,
        )
    else:
        raise NotImplementedError(f"{config.mm_projector} not support!")


class YiVLForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = YiVLConfig

    def __init__(self, config: YiVLConfig):
        LlavaPreTrainedModel.__init__(self, config)

        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = build_mm_projector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=config._attn_implementation,
            trust_remote_code=True,
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    # modify forward func due to the pixel_mask of images

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER: \nWhat's the content of the image?
        ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                if pixel_mask is None:
                    image_outputs = self.vision_tower(
                        pixel_values, output_hidden_states=False
                    )
                else:
                    image_outputs, image_atts = self.vision_tower(
                        pixel_values,
                        pixel_mask=pixel_mask,
                        output_hidden_states=False,
                    )
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.last_hidden_state

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                if pixel_mask is None:
                    image_features = self.multi_modal_projector(selected_image_feature)
                else:
                    image_features = self.multi_modal_projector(
                        selected_image_feature, attn_mask=image_atts
                    )
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                (
                    inputs_embeds,
                    attention_mask,
                    labels,
                    position_ids,
                ) = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif (
                past_key_values is not None
                and pixel_values is not None
                and input_ids.shape[1] == 1
            ):
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors
                # such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_mask=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
            }
        )
        return model_inputs

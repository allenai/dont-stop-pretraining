# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert RoBERTa checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import torch

from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from transformers.modeling_bert import (BertIntermediate, BertLayer,
                                        BertOutput,
                                        BertSelfAttention,
                                        BertSelfOutput)
from transformers.modeling_roberta import RobertaForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = 'Hello world! cécé herlolip'


def convert_roberta_checkpoint_to_pytorch(fairseq_default_path, hf_input_path):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    roberta_hf = RobertaForMaskedLM.from_pretrained(hf_input_path)
    roberta_fairseq = FairseqRobertaModel.from_pretrained(fairseq_default_path)

    # Now let's copy all the weights.
    # Embeddings
    roberta_hf_sent_encoder = roberta_hf.roberta.embeddings
    roberta_fairseq.model.decoder.sentence_encoder.embed_tokens.weight = roberta_hf_sent_encoder.word_embeddings.weight
    # fairseq roberta doesn't use `token_type_embeddings`, so as a workaround, add it to the `position_embeddings`
    roberta_fairseq.model.decoder.sentence_encoder.embed_positions.weight.data = roberta_hf_sent_encoder.position_embeddings.weight.data + roberta_hf_sent_encoder.token_type_embeddings.weight.data
    roberta_fairseq.model.decoder.sentence_encoder.emb_layer_norm.weight = roberta_hf_sent_encoder.LayerNorm.weight
    roberta_fairseq.model.decoder.sentence_encoder.emb_layer_norm.bias = roberta_hf_sent_encoder.LayerNorm.bias

    for i in range(len(roberta_hf.roberta.encoder.layer)):
        # Encoder: start of layer
        roberta_hf_layer: BertLayer = roberta_hf.roberta.encoder.layer[i]
        roberta_fairseq_layer: TransformerSentenceEncoderLayer = roberta_fairseq.model.decoder.sentence_encoder.layers[i]
        # roberta_fairseq_layer.self_attn.enable_torch_version = False

        # self attention
        hf_self_attn: BertSelfAttention = roberta_hf_layer.attention.self
        fairseq_self_attn: BertSelfAttention = roberta_fairseq_layer.self_attn
        fairseq_self_attn.q_proj.weight = hf_self_attn.query.weight
        fairseq_self_attn.q_proj.bias = hf_self_attn.query.bias
        fairseq_self_attn.k_proj.weight = hf_self_attn.key.weight
        fairseq_self_attn.k_proj.bias = hf_self_attn.key.bias
        fairseq_self_attn.v_proj.weight = hf_self_attn.value.weight
        fairseq_self_attn.v_proj.bias = hf_self_attn.value.bias

        # self-attention output
        hf_self_output: BertSelfOutput = roberta_hf_layer.attention.output
        assert(
            hf_self_output.dense.weight.shape == roberta_fairseq_layer.self_attn.out_proj.weight.shape
        )
        roberta_fairseq_layer.self_attn.out_proj.weight = hf_self_output.dense.weight
        roberta_fairseq_layer.self_attn.out_proj.bias = hf_self_output.dense.bias
        roberta_fairseq_layer.self_attn_layer_norm.weight = hf_self_output.LayerNorm.weight
        roberta_fairseq_layer.self_attn_layer_norm.bias = hf_self_output.LayerNorm.bias

        # intermediate
        hf_intermediate: BertIntermediate = roberta_hf_layer.intermediate
        assert(
            hf_intermediate.dense.weight.shape == roberta_fairseq_layer.fc1.weight.shape
        )
        roberta_fairseq_layer.fc1.weight = hf_intermediate.dense.weight
        roberta_fairseq_layer.fc1.bias = hf_intermediate.dense.bias

        # output
        hf_bert_output: BertOutput = roberta_hf_layer.output
        assert(
            hf_bert_output.dense.weight.shape == roberta_fairseq_layer.fc2.weight.shape
        )
        roberta_fairseq_layer.fc2.weight = hf_bert_output.dense.weight
        roberta_fairseq_layer.fc2.bias = hf_bert_output.dense.bias
        roberta_fairseq_layer.final_layer_norm.weight = hf_bert_output.LayerNorm.weight
        roberta_fairseq_layer.final_layer_norm.bias = hf_bert_output.LayerNorm.bias
        # end of layer

    roberta_fairseq.model.decoder.lm_head.dense.weight = roberta_hf.lm_head.dense.weight
    roberta_fairseq.model.decoder.lm_head.dense.bias = roberta_hf.lm_head.dense.bias
    roberta_fairseq.model.decoder.lm_head.layer_norm.weight = roberta_hf.lm_head.layer_norm.weight
    roberta_fairseq.model.decoder.lm_head.layer_norm.bias = roberta_hf.lm_head.layer_norm.bias
    roberta_fairseq.model.decoder.lm_head.weight = roberta_hf.lm_head.decoder.weight
    roberta_fairseq.model.decoder.lm_head.bias = roberta_hf.lm_head.bias

    # Let's check that we get the same results.
    roberta_hf.eval()  # disable dropout
    roberta_fairseq.eval()  # disable dropout
    input_ids: torch.Tensor = roberta_fairseq.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1
    our_output = roberta_hf(input_ids)[0]
    their_output = roberta_fairseq.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print(
        "Do both models output the same tensors?",
        "🔥" if success else "💩"
    )
    if not success:
        raise Exception("Something went wRoNg")

    with open(f'{fairseq_default_path}/model.pt', 'rb') as f:
        roberta_fairseq_checkpoint = torch.load(f)
    roberta_fairseq_checkpoint['model'] = roberta_fairseq.model.state_dict()
    fairseq_output_checkpoint_path = f'{hf_input_path}/fairseq.pt'
    print(f"Saving model to {fairseq_output_checkpoint_path}")
    with open(fairseq_output_checkpoint_path, 'wb') as f:
        torch.save(roberta_fairseq_checkpoint, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--fairseq_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the uncompressed fairseq roberta checkpoint (downloaded from here https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)")
    parser.add_argument("--hf_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the input huggingface PyTorch model.")
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(
        args.fairseq_path,
        args.hf_path,
    )

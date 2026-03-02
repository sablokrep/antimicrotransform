use anyhow::{Context, Result};
use anyhow::{Result, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor, VarBuilder};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, Dropout, Linear, Module, VarMap, embedding, linear, rms_norm};
use candle_nn::{Embedding, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::gpt2::{Config, Model as Gpt, RotaryEmbedding};
use clap::Parser;
use rand::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::tokenizer::{AddedToken, Tokenizer};

/*
Gaurav Sablok
codeprog@icloud.com
*/

const DNA_ALPHABET: [&str; 5] = ["A", "C", "G", "T", "N"];
const SPECIAL_TOKENS: [&str; 4] = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"];

struct AntiMicrobialClassifier {
    bert: BertModel,
    classifier: Linear,
}

impl AntiMicrobialClassifier {
    fn new(vocab_size: usize, n_classes: usize, device: &Device) -> Result<Self> {
        let config = BertConfig {
            vocab_size: vocab_size as i64,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 6,
            intermediate_size: 1536,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            ..Default::default()
        };
        let vb = VarBuilder::zeros(DType::F32, device);
        let bert = BertModel::load(&vb, &config)?;
        let classifier =
            candle_nn::linear(config.hidden_size as usize, n_classes, vb.pp("classifier"))?;

        Ok(Self { bert, classifier })
    }
    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let outputs = self.bert.forward(input_ids, token_type_ids)?;
        let pooled = outputs.pooled_output; // [batch, hidden]
        self.classifier.forward(&pooled)
    }
}

struct AnitmicrobialGpt {
    transformer: Gpt,
    lm_head: Linear,
}

impl AnitmicrobialGpt {
    fn new(vocab_size: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let transformer = Gpt::load(&vb, config)?;
        let lm_head = linear(config.n_embd, vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            transformer,
            lm_head,
        })
    }

    fn forward(&self, xs: &Tensor, pos: Option<&Tensor>) -> Result<Tensor> {
        let hidden = self.transformer.forward(xs, pos)?;
        self.lm_head.forward(&hidden)
    }
}

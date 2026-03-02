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

pub fn build_tokenizer() -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(tokenizers::models::bpe::BPE::default()));
    tokenizer.with_pre_tokenizer(Box::new(tokenizers::pre_tokenizers::char::Char::default()));
    let mut special = vec![];
    for tok in SPECIAL_TOKENS {
        special.push(AddedToken {
            content: tok.to_string(),
            single_word: true,
            lstrip: false,
            rstrip: false,
        });
    }
    tokenizer.add_special_tokens(&special);

    for base in DNA_ALPHABET {
        tokenizer.add_tokens(&[base.to_string()]);
    }

    tokenizer
}

pub fn tokenize_dna(tokenizer: &Tokenizer, seq: &str) -> Vec<u32> {
    let enc = tokenizer
        .encode(format!("[BOS]{}[EOS]", seq.to_uppercase()), true)
        .unwrap();
    enc.get_ids().to_vec()
}

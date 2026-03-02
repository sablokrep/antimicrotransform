use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::utils::precompiled::Precompiled;
use tokenizers::pre_tokenizers::char::Char;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::{AddedToken, Tokenizer};

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn build_dna_tokenizer() -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(BPE::default()));
    tokenizer.with_pre_tokenizer(Box::new(Char));
    tokenizer.add_special_tokens(&[
        AddedToken {
            content: "[PAD]".into(),
            single_word: true,
            lstrip: false,
            rstrip: false,
        },
        AddedToken {
            content: "[UNK]".into(),
            ..
        },
        AddedToken {
            content: "[CLS]".into(),
            ..
        },
        AddedToken {
            content: "[SEP]".into(),
            ..
        },
        AddedToken {
            content: "[MASK]".into(),
            ..
        },
    ]);

    // Add DNA vocab
    let bases = vec!["A", "C", "G", "T", "N"];
    for b in bases {
        tokenizer.add_tokens(&[b.to_string()]);
    }

    tokenizer
}

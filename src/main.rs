mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
use figlet_rs::FIGfont;
mod build;
mod loadfasta;
mod tokenize;
mod transform;
use crate::tokenize::build_tokenizer;
use crate::tokenize::tokenize_dna;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("beesmile");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::SMILES { filepath, thread } => {
            let n_threads = thread.parse::<usize>().expect("thread must be a number");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to create thread pool");
            pool.install(|| {
                let device = Device::cuda_if_available(0)?;
                let tokenizer = build_tokenizer();
                let vocab_size = tokenizer.get_vocab_size(true) as usize;
                let mut config = Config::gpt2_small();
                config.vocab_size = vocab_size as i64;
                config.n_positions = 1024;
                config.n_embd = 384;
                config.n_layer = 6;
                config.n_head = 6;
                config.intermediate_size = Some(1536);
                let mut varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
                let model = DnaGpt::new(vocab_size, &config, vb)?;
                println!("Model ready ({} params approx)", varmap.all_vars().len());
                let mut logits_proc =
                    LogitsProcessor::new(Some(args.temperature as f64), None, None);
                for i in 0..args.num_gen {
                    let prompt = "[BOS]".to_string();
                    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
                    print!("GEN {}: ", i + 1);
                    for _ in 0..args.gen_len {
                        let context_len = tokens.len().min(config.n_positions as usize);
                        let input_ids = Tensor::from_vec(
                            tokens[tokens.len() - context_len..].to_vec(),
                            (1, context_len),
                            &device,
                        )?;
                        let logits = model.forward(&input_ids, None)?.i((0, D::Minus1))?;
                        let logits = logits_proc.sample(&logits, args.top_k)?;
                        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
                        if let Some(tok_str) = tokenizer.id_to_token(next_token) {
                            if tok_str == "[EOS]" {
                                break;
                            }
                            print!("{}", tok_str);
                        }
                        tokens.push(next_token);
                    }
                    println!();
                }
            });
        }
    }
}

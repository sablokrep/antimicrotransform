use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "AntiMicrobialTransform",
    version = "1.0",
    about = "Antimicrobial Transformers using Candle
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// build the graph
    AntiMicrobialTransformer {
        /// path to the file
        filepath: String,
        /// threads for the analysis
        thread: String,
        /// generate sequences after training
        generate: bool,
        /// number of sequences to generate
        numgen: usize,
        /// max length of generate sequences
        genelength: usize,
        /// temperature for samples
        temperature: f64,
        /// top k for sampling
        topk: usize,
    },
}

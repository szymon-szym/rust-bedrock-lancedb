#[derive(clap::Parser, Debug)]
pub struct Config {
    #[clap(long, env)]
    pub(crate) bucket_name: String,
    #[clap(long, env)]
    pub(crate) prefix: String,
    #[clap(long, env)]
    pub(crate) table_name: String,
}
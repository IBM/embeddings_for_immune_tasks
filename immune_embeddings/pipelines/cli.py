import click

from immune_embeddings.utils import load_config


@click.command(name='run_experiment')
@click.option("--config", "config_path", help="path to the config file", type=str)
def exp_cli(config_path):
    config = load_config(config_path)
    from immune_embeddings.pipelines.experiment import run_experiment
    run_experiment(config)

from typing import Dict, Any


def get_model_config() -> Dict[str, Any]:
    embedding_size = 64
    representation_rnn_hidden_size = 128

    return {
        "embedding_size": embedding_size,
        "representation_rnn": {
            "input_size": embedding_size,
            "hidden_size": representation_rnn_hidden_size,
            "num_layers": 1
        },
        "command_scorer_net": [
            representation_rnn_hidden_size,
            64]
    }


config = {
    "general": {
        "game_files": [
            "data/SaladWorld/level_1.ulx",
            # "data/SaladWorld/level_2.ulx",
            # "data/SaladWorld/level_3.ulx"
        ],
        "commands_files": [
            "conbas/agents/lstm_dqn/commands/cmd_saladworld_level_1.txt",
            # "conbas/agents/lstm_dqn/commands/cmd_saladworld_level_2.txt",
            # "conbas/agents/lstm_dqn/commands/cmd_saladworld_level_3.txt",
        ],
        "vocab_file": "conbas/agents/lstm_dqn/vocab.txt",
        "linear_anneald_args": {
            "start_eps": 1.0,
            "end_eps": 0.05,
            "duration": 12_000
        }
    },
    "training": {
        "n_episodes": 16_000,
        "batch_size": 16,
        "max_steps_per_episode": 50,
        "loss_fn": "smooth_l1",
        "replay_batch_size": 16,
        "replay_capacity": 30_000,
        "update_after": 100,
        "soft_update_tau": 0.001,
        "discount": 0.99,
        "optimizer": {
            "lr": 0.001,
            "clip_grad_norm": 5.
        }
    },
    "model": get_model_config(),
    "checkpoint": {
        "on_exist": "delete",
        "experiments_path": "experiments/",
        "experiment_tag": "lstm_dqn",
        "save_frequency": 10,
    }
}

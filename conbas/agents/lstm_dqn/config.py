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
        "vocab_file": "conbas/agents/lstm_dqn/vocab.txt"
    },
    "training": {
        "replay_batch_size": 16,
        "replay_capacity": 30_000,
        "update_after": 100,
        "n_episodes": 16_000,
        "n_epochs": 100,
        "batch_size": 16,
        "max_steps_per_episode": 50,
        "soft_update_tau": 0.001,
        "discount": 0.99,
        "optimizer": {
            "lr": 0.001,
            "clip_grad_norm": 5.
        }
    },
    "model": {
        "embedding_size": 4,
        "hidden_size": 4
    }
}

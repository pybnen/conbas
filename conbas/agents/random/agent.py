from typing import Optional, List, Dict, Any
import numpy as np

from textworld import EnvInfos


class RandomAgent:

    def __init__(self, commands: Optional[List[str]] = None, use_admissible=False) -> None:
        assert commands or use_admissible

        self.commands = commands
        self.use_admissible = use_admissible

    def request_infos(self) -> Optional[EnvInfos]:
        """Request the infos the agent expects from the environment

        Returns:
            request_infos: EnvInfos"""
        request_infos = EnvInfos()
        request_infos.admissible_commands = self.use_admissible
        return request_infos

    def init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        pass

    def act(self, obs: List[str], infos: Dict[str, List[Any]]
            ) -> List[str]:
        if self.use_admissible:
            return [str(np.random.choice(commands)) for commands in infos["admissible_commands"]]
        else:
            return np.random.choice(self.commands, size=len(obs)).tolist()

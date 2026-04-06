import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

from models import EdaOpenenvAction, EdaOpenenvObservation
from server.EDA_OpenEnv_environment import EdaOpenenvEnvironment


def env_factory() -> EdaOpenenvEnvironment:
    """Factory function — create_app calls this to create new env instances."""
    return EdaOpenenvEnvironment()


app = create_app(
    env=env_factory,
    action_cls=EdaOpenenvAction,
    observation_cls=EdaOpenenvObservation,
    env_name="EDA_OpenEnv",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=8000)
    # args = parser.parse_args()
    # main(port=args.port)
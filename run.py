import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Autonomous Chicken Invaders Pilot")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: record
    record_parser = subparsers.add_parser("record", help="Record human gameplay demonstrations on Stage 1")
    record_parser.add_argument("--preview", action="store_true", help="Show OpenCV debug preview window")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train Behavioral Cloning policy on recorded demonstration")
    train_parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Command: play
    play_parser = subparsers.add_parser("play", help="Launch autonomous autopilot engine")
    play_parser.add_argument("--no-policy", action="store_true", help="Run with heuristic pilot instead of trained neural network")
    play_parser.add_argument("--pure-model", action="store_true", help="Give trained neural model 100 percent direct control without safety dodger overrides")
    play_parser.add_argument("--fps", type=float, default=100.0, help="Target framerate (default: 100.0 FPS)")
    play_parser.add_argument("--sync", action="store_true", help="Run synchronous YOLO inference instead of 100+ FPS asynchronous worker")
    play_parser.add_argument("--hud", action="store_true", help="Show OpenCV visual HUD window")

    # Command: rl
    rl_parser = subparsers.add_parser("rl", help="Run Reinforcement Learning (PPO) self-training loop")
    rl_parser.add_argument("--iterations", type=int, default=30, help="Total PPO training iterations")
    rl_parser.add_argument("--steps", type=int, default=100, help="Steps per rollout")

    args = parser.parse_args()

    if args.command == "record":
        from src.recorder import DemonstrationRecorder
        recorder = DemonstrationRecorder(show_preview=args.preview)
        recorder.run()

    elif args.command == "train":
        from src.train_bc import train_behavioral_cloning
        train_behavioral_cloning(epochs=args.epochs, batch_size=args.batch_size)

    elif args.command == "play":
        from src.engine import ChickenPilotEngine
        engine = ChickenPilotEngine(
            use_policy_model=not args.no_policy,
            show_hud=args.hud,
            pure_model=args.pure_model,
            target_fps=args.fps,
            async_inference=not args.sync,
        )
        engine.run()


    elif args.command == "rl":
        from src.environment import ChickenInvadersEnv
        from src.train_rl import PPOTrainer
        env = ChickenInvadersEnv()
        trainer = PPOTrainer(env)
        try:
            trainer.train(total_iterations=args.iterations, steps_per_rollout=args.steps)
        finally:
            env.close()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

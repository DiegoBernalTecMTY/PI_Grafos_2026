"""
quick_train_w2v.py — 10-epoch smoke-test + full group MRR evaluation.

Identical to train_ikge_w2v.py in every way except:
  • epochs     = 10   (instead of 200)
  • eval_every = 1    (val loss printed every epoch)
  • run_name   = "quick10" (log file labelled distinctly)

Usage:
    python quick_train_w2v.py

To load a previously saved checkpoint and skip straight to evaluation, use
eval_from_checkpoint_w2v.py instead.
"""

import argparse
from train_ikge_w2v import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="10-epoch quick train + MRR group evaluation (IKGE Wikipedia2Vec)"
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of training triples to use (default: 1.0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Validate every N epochs (default: 1, shows loss gap every epoch)"
    )
    args = parser.parse_args()

    main(
        fraction=args.fraction,
        run_name="quick10",
        epochs=args.epochs,
        eval_every=args.eval_every,
    )

import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n=======================================================")
    print(f"RUNNING: {cmd}")
    print(f"=======================================================\n")
    # Stream output to console
    process = subprocess.Popen(cmd, shell=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Error running {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    commands = [
        "python training/train.py --model segformer --epochs 50 --batch_size 8 --seed 123",
        "python training/train.py --model deeplabv3 --epochs 50 --batch_size 8 --seed 456",
        "python training/train_prithvi.py --epochs 50 --batch_size 2 --seed 42 --lora_rank 32 --unfreeze_blocks 0"
    ]
    
    for cmd in commands:
        run_cmd(cmd)
    
    print("\n✅ ALL MODELS HAVE BEEN RETRAINED ON THE ALIGNED 2021 DATASET!")

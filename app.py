import os
os.system("sudo apt update")
os.system("sudo apt install --only-upgrade sqlite3")  # for Chroma DB
os.system(f"git lfs install")
os.system(f"pip install torch torchvision torchaudio torchtext torchdata --extra-index-url https://download.pytorch.org/whl/cu118 -U")
os.system(f"pip install bitsandbytes sentencepiece fsspec gradio langchain einops scipy unstructured xformers sentence-transformers chromadb -U")
os.system(f"pip install git+https://github.com/huggingface/transformers.git -U")
os.system(f"pip install git+https://github.com/huggingface/accelerate.git -U")
os.system(f"pip install pydantic==1.10.11")
os.system(f"cp .venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda118.so .venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so")
os.system(f"python run.py")
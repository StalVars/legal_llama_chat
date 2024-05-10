git remote add origin git@github.com:StalVars/legal_llama_chat.git
git branch -M main
git push -u origin main


#Installation
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt


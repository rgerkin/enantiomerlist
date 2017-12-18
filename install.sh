export PATH=$PATH:$HOME/anaconda3_420/bin
conda install -y -c rdkit -c mordred-descriptor mordred
wget https://github.com/dream-olfaction/olfaction-prediction/archive/thin.zip
unzip thin.zip
mv olfaction-prediction-thin olfaction-prediction

#Install the dependencies 
!pip install kaggle
#Upload Kaggle Api Token 
from google.colab import files
files.upload() #upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d mksaad/arabic-sentiment-twitter-corpus
#unzip the data 
!unzip -q arabic-sentiment-twitter-corpus.zip -d .

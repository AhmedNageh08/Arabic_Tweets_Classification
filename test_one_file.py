#function that does prediction on text file 
#each tweet is in one line
#it takes data_file path & model_file path
#and print out the result  
import pickle
def predict_file(data_file,model_file):
   with open(data_file, 'r') as f:
      corpus = [line.strip() for line in f]
   with open(model_file, 'rb') as file:
      pickle_model = pickle.load(file)
   result = [pickle_model.predict([d]) for d in corpus]
   values, counts = np.unique(result, return_counts=True)
   neg_per = (counts[0] / len(result)) * 100
   pos_per = (counts[1] / len(result)) * 100
   print(f"The text has {counts[0]} negative sentences by perctenage {neg_per:.2f} %" )
   print(f"The test has {counts[1]} positive sentences by percentage {pos_per:.2f} %")
   
   
#calling the prediction function
predict_file("/content/test.txt","/content/drive/MyDrive/LinearSVC_1.pkl")

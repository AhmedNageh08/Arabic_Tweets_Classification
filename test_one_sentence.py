#Predict using one tweet
text =["هي وصلت لدرجادي 😯 لا الموضوع كدا محتاج وقفه 💔😂😂"]
#Import the saved file of the model
pkl_filename = "/content/drive/My Drive/SVC_2.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

result = pickle_model.predict(text)
print(result)

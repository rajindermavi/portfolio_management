from sentence_transformers import SentenceTransformer
from singleton_decorator import singleton

@singleton
class NLEncoding(): 
  
    def __init__(self):  
        '''Initialize the Natural Language Encoder with a pretrained encoder.'''
        pretrained = 'paraphrase-distilroberta-base-v1'
        self.model = SentenceTransformer(pretrained)
        
    def vectorize(self,text):
        
        return self.model.encode(text)
         
    def vectorize_news_series(news_series):
        '''Takes in news series and converts it to sequence of numerical vectors'''
      
        nle = data_processing.NLEncoding()
        vec = nle.vectorize(news_series)
        
        return pd.DataFrame(data = vec,index = news_series.index) 
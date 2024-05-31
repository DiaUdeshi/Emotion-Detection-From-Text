import pandas as pd
import neattext.functions as nfx

d1 = pd.read_csv('tweet_emotions.csv')
d2 = pd.read_csv('emotion_data.csv')
d3 = pd.read_csv('emotion_to_text.csv')

concat_df = pd.concat([d1[['sentiment','text']], d2[['sentiment','text']], d3[['sentiment','text']]], ignore_index=True)
concat_df['text']=concat_df['text'].apply(nfx.remove_userhandles)
concat_df.to_csv('final.csv',index=False)


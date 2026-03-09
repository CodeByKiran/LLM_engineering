# What is the Difference between a token and a Word ? Provide Examples from the HomeNest Dataset Where they differ significantly .
 Word -> A Lingustic unit , a word seperated by spaces.
 Token -> A unit that NLP Model Processes (Token are produced by a process of Tokenization)
 Tokens can be words ,sub words , punctuation ,special symbols

 eg : unbelievable ( words -1 | tokens - 3 [un,believe,able]) 
 All Words Can be TOKENS but all tokens can't be Words.

 eg from HomeNest Dataset : 

1)The MirrorPro LED Vanity Mirror is easy to clean, quiet during operation, and looks great. 5 stars!
  Tokens : 22 [The Mirror Pro LED Vanity Mirror is easy to clean , quiet during operation , and looks great . 5 stars !]
  Words  : 17 [The MirrorPro led vanity mirror is easy to clean quiet duirng operation and looks great 5 stars]

# Why would you set temperature=0.0 for sentiment analysis but temperature=0.7 forsummarization?  
 Temperature -> Randomness & Creativity Dial
 By increasing the Temperature we also increase the probability distribution of the next word 

 for sentiment analysis , we don't need to generate any new text , we analyze the existing text and classify it as +ve , -ve or neutral 
 whereas in Text Summarization , the objective is to summarize the given text by preserving the Context of the original review.


 # What is the risk of using max_tokens=50 for summarization? What aboutmax_tokens=2000? 

  The Model can generate the max_token specified , the risk of using max_tokens =50 is it may turnate the Summaries and max_tokens = 2000 may generate long responses 
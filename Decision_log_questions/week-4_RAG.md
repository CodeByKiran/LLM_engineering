# For Question 10 (return policy): did your system hallucinate? If yes, what exact change tothe system prompt fixed it? Paste both versions.
 No , it's response was - "Based on the provided reviews, there is **no clear information** about HomeNest’s official return policy. However, multiple customers report difficulties with returns:

- **CloudPillow Set (2-pack)**: One customer (#1974) mentions a ""nightmare"" trying to return the product due to no response from customer service.
- **Damaged/Defective Items**: Several customers (#2847, #2802, #107) attempted returns for damaged or chemical-smelling products, but the reviews do not specify the policy details.

For exact return policy terms, additional context is needed.","[1974, 4525, 2847, 2802, 107]"


# Compare your custom RAG answer quality vs LangChain RAG on 3 questions. Are they thesame? If different, why?
No - they are not exactly the Same 

Custom RAG is more direct and uses raw retrived Chunks

langchain RAG gives more Structured and polished answer


eg:
CUSTOM RAG

What are the most common complaints about kitchen appliances?
"Based on the provided reviews, the most common complaints about kitchen appliances are:

1. **Quality issues** – Customers expected better quality for products like the **SteamKing Rice Cooker** (#3602) and **CrispAir Fryer XL** (#2124, #3586).
2. **Noise problems** – The **CrispAir Fryer XL** (#2221) has a loud grinding motor.
3. **Safety hazards** – The **SliceMaster Food Processor** (#297) sparked upon use, posing a serious risk.","[3602, 2221, 2124, 3586, 297]"



Langchain RAG

What are the most common complaints about kitchen appliances?
ANSWER: Based on the provided customer reviews, the most common complaints about kitchen appliances are:

1. **Quality Issues** – Multiple reviews mention that the CrispAir Fryer XL is of "average" or "nothing special" quality, and one explicitly states it was "fine" but expected better. Another review highlights a defective SteamKing Rice Cooker.

2. **Defective Products** – One customer received a defective SteamKing Rice Cooker, and another had the wrong item sent twice before receiving the correct (but defective) product.

3. **Mixed Performance** – While some appliances (like the TurboBlend Pro 5000) are praised, others (like the CrispAir Fryer XL) are described as "okay" or "decent for the price," suggesting they meet basic expectations but lack standout features.

4. **Shipping/Logistics Problems** – A customer had the wrong item sent twice, leading to frustration.

The most recurring complaints revolve around **quality, defects, and shipping errors**, while performance is generally seen as average rather than exceptional.


# What is the 'stuff' chain type in LangChain? What are the alternatives (map_reduce, refine)?When would you use each?
The stuff Chain simply  puts all the retrieved documents together in one prompt and send it to the LLM

           Retrieve documents → Concatenate them → Send to LLM → Generate answer

| Chain Type | How It Works                           | Pros                   | Cons                        | Best Use Case     |
| ---------- | -------------------------------------- | ---------------------- | --------------------------- | ----------------- |
| Stuff      | All docs sent in one prompt            | Fast, simple           | Token limits, noisy context | Small datasets    |
| map_reduce | Process docs individually then combine | Scales to many docs    | May lose detail             | Large corpora     |
| refine     | Iteratively improves answer            | High-quality reasoning | Slow and costly             | Detailed analysis |



#  Your RAG retrieves top_k=5 reviews. What happens when you set top_k=1? top_k=15?What is the trade-off between context richness and token cost?

top_k determines how many documents are retrieved and sent to the LLM.

🔹 top_k = 1 
(Pros)

Very low token cost 

Fast response 

High precision

(Cons)

Limited context

May miss important information

Answers may be incomplete



🔹 top_k = 5 (Balanced Setting)

(Pros)

Good balance of context and cost

Captures multiple relevant reviews

More reliable answers

(Cons)

Slightly higher token usage than k=1



🔹 top_k = 15

(Pros)

Richer context

Higher chance of capturing all relevant information

(Cons)

More irrelevant documents (noise)

Higher token cost 

Slower responses


Trade-off

Factor	          |  Low top_k |	High top_k
Context richness  |	  Low	   |    High
Precision	      |    High	   |   Lower
Token cost	      |    Low	   |    High
Noise risk	      |     Low	   |   High


Small top_k → precise but incomplete answers,

Large top_k → richer context but noisy and expensive

In many RAG systems, top_k ≈ 5 provides the best balance.
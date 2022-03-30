# Document Query - Cosine Similarity Workshop

</br>

## Objective

</br>

To practice using Cosine Similarity as a tool for performing searching for relevant documents based on a given query string

</br>

## Your Task

</br>

- You have a text file called “quotes.txt” that contains 51 quotes. 

</br>

- Given a query string, list out the most relevant quotes.

  - First, treat each quote as a separate document and compute TF-IDF values for each word in the 51 documents (remember to treat each line of quote as a separate document; your corpus is the 51 documents).
  
  </br>

  - Next, treat the query string as a separate document and compute its TF-IDF values as well.
  
  </br>

  - Then, perform Cosine Similarity between the query string (treat it as a document) and the 51 documents.

  </br>

  - A Cosine Similarity value that is greater than 0 is regarded as a relevant document for the query string. Rank the documents (i.e. lines of quote) by their Cosine Similarity scores in descending order (higher scores first)

  </br>

- Use the query string – “life wise choices” to test your results.

</br>

- Given the above query string, your results should be:

  >Query: life wise choices\
  Results:\
  "Life is the sum of our choices." [score:0.7393218972723493]\
  \
  "It is our choices that show what we really are, far more than our abilities." [score:0.35320464597221546]\
  \
  "You attract into your life that which you are." [score:0.32095071358388955]\
  \
  "Challenges are what make life interesting and overcoming them is what makes life meaningful." [score:0.24935776599022805]\
  \
  "Let your life be shaped by decisions you made, not by the ones you didn't." [score:0.15295031439879156]\
  \
  "You can live your whole life and never know who you are; until you see the world through the eyes of others." [score:0.1427887948846313]\
  \
  "Our greatest fear should not be of failure, but of succeeding at things in life that don't really matter." [score:0.13006308610420023]\
  \
  "Don't be afraid your life will end; be afraid that it will never begin." [score:0.1290320332985028]\
  \
  "There are two things to aim at in life: first, to get what you want; and, after that, to enjoy it. Only the wisest of mankind achieve the second." [score:0.11289841969902241]\
  \
  "To see the world, things dangerous to come to. To see behind walls, to draw closer. To find each other and to feel. That, is the purpose of life." [score:0.10171860873681889]

</br>

## Reminder

</br>

- For pre-processing, remember to remove stop-words from your corpus and stem each word in the text.












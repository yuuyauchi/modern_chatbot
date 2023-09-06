import wikipedia

wikipedia.set_lang("jp")
import pdb;pdb.set_trace()
word="リオネルメッシ"
words = wikipedia.search(word)
wikipedia.summary(words[0])
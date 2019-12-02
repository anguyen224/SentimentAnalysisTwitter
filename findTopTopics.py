import collections
c = ['again i am so so sorry this has happened my heart is with you feel free to share thoughts to me if you like', 'yumm bacon and srambled eggs', 'ooh gosssh i wont do anything this envening for the first timee i will learn my stupid brevet', 'ugh i wish i wasnt at work so i could get some moes right now', 'you pretty much made ich explode were proud though and honored', 'i know it is fast approaching we should have a little celebration that day', 'hangover free', 'i tweeted someone asking if obrian cleaved jays desk in twain and made off with some wenches dont think he got the joke', 'i should probably start practicing physics over the summer so i dont get rusty after all of the universe is still missing', 'concert please explain im intrigued']

def find_top_topics(corpus):
    topics = {}
    for sentence in corpus:
        for word in sentence.split():
            if word not in topics:
                topics[word] = 1
            else: topics[word] +=1
    d = collections.Counter(topics)
    return d.most_common(5)
find_top_topics(c)

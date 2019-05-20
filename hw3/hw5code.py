import math, random


################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    ngrams=[]
    text = start_pad(n)+text

    for i in range(len(text)-n):
        ngrams.append((text[i:i+n], text[i+n]))
    return ngrams


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.vocab= []
        self.n=n
        self.k=k
        self.countContext={}
        self.countContextChar={}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            if char not in self.vocab:
                self.vocab.append(char)

        ngram= ngrams(self.n, text)

        for (cnt, v) in ngram:
            if cnt not in self.countContext.keys():
                self.countContext[cnt] = 1
            else:
                self.countContext[cnt] += 1
        
        for (cnt, v) in ngram:
            if (cnt, v) not in self.countContextChar.keys():
                self.countContextChar[(cnt,v)] = 1
            else:
                self.countContextChar[(cnt,v)] += 1



                
    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        proba =1
        if (context,char) in self.countContextChar.keys():
            proba = (self.countContextChar[(context, char)] + self.k)/(self.countContext[context] + self.k*len(self.vocab))
        elif (context, char) not in self.countContextChar.keys() and context in self.countContext.keys():
            proba = (self.k)/(self.countContext[context] + self.k*len(self.vocab))
        else:
            proba= 1/len(self.vocab)
        return proba

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        vocab= sorted(self.vocab)
        r=random.random()
        for i in range(0, len(vocab)):
            acumProb1 = 0
            acumProb2 = 0
            for j in range(i):
                acumProb1 += self.prob(context, vocab[j])
            for j in range(i+1):
                acumProb2 += self.prob(context, vocab[j])
            if acumProb1 <= r and acumProb2 > r:
                return vocab[i]


    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        icontex = start_pad(self.n)
        text=''
        for i in range(length):
            char = self.random_char(icontex)
            text+=char
            icontex= icontex[1:] + char
        return text



    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        #there is an update after the 
        icontex = start_pad(self.n)
        perplexity=0

        for i in range(len(text)):
            char = text[i]
            if self.prob(icontex, char)==0:
                return float('inf')
            perplexity += -math.log(self.prob(icontex, char))
            icontex = icontex[1:] + char
        return math.pow(math.exp(perplexity),1/len(text))

        
################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.vocab = []
        self.n=n
        self.k=k
        #self.countContext={}
        #self.countContextChar={}
        self.models=[]
        self.landas=[]
        for i in range(self.n+1):
            self.landas.append(1/(self.n+1))
        for i in range(self.n+1):
            self.models.append(NgramModel(i,self.k))

    def updateLandas(self, arrayLandas):
        self.landas =arrayLandas
        #for i in range(len(arrayLandas)):
        #    self.landas[i]=arrayLandas[i]

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        for i in self.models:
            i.update(text)
    
    def prob(self, context, char):
        probs=0
        for i in range(self.n+1):
            m =self.models[i]
            probs+=m.prob(context[self.n-i:], char)*self.landas[i]
        return probs



################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':

    #------------PART 0----------#
    print('\n#------------PART 0----------#')
    print(ngrams(1, 'abc'))
    print( ngrams(2, 'abc'))

    #------------PART 1----------#
    print('\n#------------PART 1----------#')
    #POINT 2
    print('\npoint2')
    m = NgramModel(1, 0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print( m.prob('a', 'b'))
    print(m.prob('~', 'c'))
    print(m.prob('b', 'c'))

    #POINT 3
    print('\npoint3')
    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print([m.random_char('') for i in range(25)])

    #POINT 4
    print('\npoint4')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print(m.random_text(25))

    #WILLIAM SHAKESPEARE TEXT
    print('\nwilliam shakespeare using different n')

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    random.seed(1)
    print('\nn=2------------------------------\n', m.random_text(250))

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    random.seed(1)
    print('\nn=3------------------------------\n', m.random_text(250))
  
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    random.seed(1)
    print('\nn=4------------------------------\n', m.random_text(250))

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    random.seed(1)
    print('\nn=7------------------------------\n', m.random_text(250))
    
    
    #------------PART 2----------#
    print('\n#------------PART 2----------#')

    #PERPLEXITY
    print('\n#PERPLEXITY')
    
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
    print(m.perplexity('abca'))
    print(m.perplexity('abcda'))

    filename1 = 'shakespeare_input.txt'
    filename2 = 'shakespeare_sonnets.txt'
    filename3 = 'nytimes_article.txt'

    text2 = open(filename2, encoding='utf-8', errors='ignore').read()
    text3 = open(filename3, encoding='utf-8', errors='ignore').read()

    t2 = []
    for i in range(0, len(text2), 100):
        t2.append(text2[i:i+100])
    t3 = []
    for i in range(0, len(text3), 100):
        t3.append(text3[i:i+100])

    #----n=3
    print('\n#----n=3, k=1')
    m = create_ngram_model_lines(NgramModel, filename1, 3, 1)

    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)

 

    #SMOOTHING
    print('\n#SMOOTHING')

    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('a', 'a'))
    print(m.prob('a', 'b'))
    print(m.prob('c', 'd'))
    print(m.prob('d', 'a'))

    #INTERPOLATION
    print('\n#INTERPOLATION')

    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print(m.prob('a', 'a'))
    m.prob('a', 'b')
    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('~a', 'b'))
    print(m.prob('ba', 'b'))
    print(m.prob('~c', 'd'))
    print(m.prob('bc', 'd'))

    #PERPLEXITY using different smothings
    print('\nPERPLEXITY')

    filename1 = 'shakespeare_input.txt'
    filename2 = 'shakespeare_sonnets.txt'
    filename3 = 'nytimes_article.txt'

    text2 = open(filename2, encoding='utf-8', errors='ignore').read()
    text3 = open(filename3, encoding='utf-8', errors='ignore').read()

    t2 = []
    for i in range(0, len(text2), 100):
        t2.append(text2[i:i+100])
    t3 = []
    for i in range(0, len(text3), 100):
        t3.append(text3[i:i+100])


    print('\nPERPLEXITY using different K smothings and interpolation')

    #----K=1 and n=3
    print('\n#----K=1 and n=3')
    m = create_ngram_model_lines(NgramModelWithInterpolation, filename1, 3, 1)

    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)
    
    #----K=10 and n=3
    print('\n#----K=10 and n=3')
    m = create_ngram_model_lines(NgramModelWithInterpolation, filename1, 3, 10)

    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)
    
    #----K=100 and n=3
    print('\n#----K=100 and n=3')
    m = create_ngram_model_lines(NgramModelWithInterpolation, filename1, 3, 100)

    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)


    #PERPLEXITY using different lamdas
    print('\nPERPLEXITY using different interpolation and diferent lambas with K=1 and n=3')

    m = create_ngram_model_lines(NgramModelWithInterpolation, filename1, 3, 1)
    
    #landas ecual weight [0.25,0.25,0.25,0.25]
    print('\n#---------------------lamdas:', m.landas)

    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)

    #landas update to [0.05,0.15,0.3,0.5]
    m.updateLandas([0.05, 0.15, 0.3, 0.5])
    print('\n#----------------------lamdas:', m.landas)
    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)

    #landas update to [0.01,0.09,0.2,0.7]
    m.updateLandas([0.01, 0.09, 0.2, 0.7])
    print('\n#----------------------lamdas:', m.landas)
    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)

    #landas update to [0.7,0.2,0.09,0.01]
    m.updateLandas([0.7, 0.2, 0.09, 0.01])
    print('\n#----------------------lamdas:', m.landas)
    #-------------shakespeare_sonnets.txt
    print('\n#-------------shakespeare_sonnets.txt')
    per = 0
    for i in t2:
        per += m.perplexity(i)
    per = per/len(t2)
    print('perplexity: ', per)

    #-------------nytimes_article.txt
    print('\n#-------------nytimes_article.txt')
    per = 0
    for i in t3:
        per += m.perplexity(i)
    per = per/len(t3)
    print('perplexity: ', per)


    

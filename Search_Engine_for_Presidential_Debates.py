#
# Name     : Sanjay Thapa
# Course   : Data Mining (CSE 5334)
# Due date : October 01, 2018
#

"""
 Purpose: The program will read a corpus and produce TF-IDF vectors for documents in the corpus, and
          given a query string, the code will return the query answer: the document with the highest
          cosine similarity score for the query.

          The program will implement a smarter threshold-bounding algorithm instead of computing cosine similarity score for each and every document.
"""

import os
import math
from collections import Counter

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

dictionary_for_documentNum_and_finalTokens = {}

dictionary_for_documentNum_and_Count_of_words = {}

dict_for_tf_idf_in_normalized_form_for_all_documents = {}

# The directory of the folder having all the presidential debates
corpusroot = 'presidential_debates'

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

total_number_of_documents = 0

def create_dictionary_for_documentNum_and_finalTokens_and_Count_function():

    global total_number_of_documents
    total_number_of_documents = 0

    global dictionary_for_documentNum_and_Count_of_words  # Added global
    global dictionary_for_documentNum_and_Count_of_words  # added global

    # Clear the dictionary to make it empty before starting to call function
    dictionary_for_documentNum_and_Count_of_words.clear()

    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        doc = doc.lower()

        final_tokens_for_each_document = create_final_tokens_after_removing_stopwords_and_stemming_function(doc)

        total_number_of_documents = total_number_of_documents + 1

        dictionary_for_documentNum_and_finalTokens[filename] = final_tokens_for_each_document
        dictionary_for_documentNum_and_Count_of_words[filename] = Counter(final_tokens_for_each_document)

        file.close()


# This function creates token after removing stopwords and stemming all words fro corpus
def create_final_tokens_after_removing_stopwords_and_stemming_function(debate_document_number):

    token_list_after_implementing_stopwords_and_stemming = []

    # Tokenize the document number passed in the function
    tokens_of_document = tokenizer.tokenize(debate_document_number)

    # For getting all the stopwords in English
    stopWords_in_English = []
    stopWords_in_English = stopwords.words('english')

    # This loop will remove any stopwords in the document and afterwards stems all the tokens
    for each_token in tokens_of_document:
        if each_token not in stopWords_in_English:
            each_token = stemmer.stem(each_token)
            token_list_after_implementing_stopwords_and_stemming.append(each_token)

    return token_list_after_implementing_stopwords_and_stemming


# return a tuple in the form (filename of the document, score), where the document is the query answer with to "qstring"
# This does not return the correct document so commented for now. Will fix in the next version.
#def query(qstring):
#
#
#    qstring = qstring.lower()
#
#    qstring = create_final_tokens_after_removing_stopwords_and_stemming_function(qstring)
#
#    dict_for_storing_norm_tf_of_qstring = {}
#
#    dict_for_qstring_as_Counter = Counter(qstring)
#
#    total_sum = 0
#    normalized_total_sum = 0.0
#
#    # this loop will find the normalized value in order to calculate term frequency
#    for key,values in dict_for_qstring_as_Counter.items():
#        total_sum = total_sum + (values * values)
#
#    normalized_total_sum = math.sqrt(total_sum)
#
#    # this loop will insert the respective normalized  tf of the queries
#    for key2, values2 in dict_for_qstring_as_Counter.items():
#        normalized_value_of_key = 0
#        normalized_value_of_key = (1 + math.log10(values2))/normalized_total_sum
#        dict_for_storing_norm_tf_of_qstring[key2] = normalized_value_of_key
#
#    return ("None",0.0)

# The function will return inverse document frequency of the the passed token in the function
def getidf(token):
    global total_number_of_documents
    global dictionary_for_documentNum_and_Count_of_words # Added this line to make global

    count_the_number_of_documents_with_the_given_token = 0
    # create a dictionary to store the tokens and its count in the Count object

    dictionary_for_token_and_its_count_in_each_document = {}

    for key, values in dictionary_for_documentNum_and_Count_of_words.items():
        # print(key)    # Delete this code later

        dictionary_for_token_and_its_count_in_each_document = values
        if dictionary_for_token_and_its_count_in_each_document[token] == 0:
            pass
        else:
            count_the_number_of_documents_with_the_given_token =\
                            count_the_number_of_documents_with_the_given_token + 1


    number_of_documents_with_term_t = 0
    number_of_documents_with_term_t = count_the_number_of_documents_with_the_given_token

    # Check if the given documents contain the token passed in getidf(token) function
    if number_of_documents_with_term_t == 0:

        return -1

    else:
        # Calculating the inverse document frequency for the given token
        inverse_document_frequency_weight = 0

        inverse_document_frequency_weight = math.log10(total_number_of_documents / number_of_documents_with_term_t)

        return inverse_document_frequency_weight

# This function will calculate the TF-IDF vectors of each document
def calculate_tf_idf_in_normalized_form_function():

    global dict_for_tf_idf_in_normalized_form_for_all_documents
    dict_for_doc_num_and_tf_idf = {}
    # counter_dict_for_each_document = {}
    dict_for_only_tf = {}
    dict_for_tf_idf_vector_of_each_document = {}

    dict_for_tf_idf_in_normalized_form = {}

    term_frequency = 0
    inverse_document_frequency = 0

    for keys, values in dictionary_for_documentNum_and_Count_of_words.items():
        counter_dict_for_each_document = {}
        counter_dict_for_each_document = Counter(values)


        for counter_keys, counter_tf_values in counter_dict_for_each_document.items():
            term_frequency = 0
            term_frequency = 1 + math.log10(counter_tf_values)
            #print(term_frequency)
            inverse_document_frequency = getidf(counter_keys)
            dict_for_tf_idf_vector_of_each_document[counter_keys] = term_frequency * inverse_document_frequency

        # dict_for_tf_idf_vector_of_each_document[keys] = dict_for_only_tf

        square_sum = 0
        square_root_of_square_sum = 0

        for keys2, values2 in dict_for_tf_idf_vector_of_each_document.items():
            square_sum = square_sum + (values2 * values2)

        square_root_of_square_sum = math.sqrt(square_sum)

        tf_idf_normalized_value = 0
        for keys3, values3 in dict_for_tf_idf_vector_of_each_document.items():
            tf_idf_normalized_value = values3/square_root_of_square_sum
            dict_for_tf_idf_in_normalized_form[keys3] = tf_idf_normalized_value

        dict_for_tf_idf_vector_of_each_document = {}   # make the dictionary empty
        dict_for_tf_idf_in_normalized_form_for_all_documents[keys] = dict_for_tf_idf_in_normalized_form
        dict_for_tf_idf_in_normalized_form = {}        # make the dictionary empty to use it again

# This function calculates the normalized tf-idf for the given token in the given filename
def getweight(filename,token):

    dict_to_get_tf_idf = {}
    value_of_tf_idf_in_doc = 0.0

    global dict_for_tf_idf_in_normalized_form_for_all_documents
    dict_to_get_tf_idf = dict_for_tf_idf_in_normalized_form_for_all_documents[filename]

    try:
        value_of_tf_idf_in_doc = dict_to_get_tf_idf[token]
    except KeyError:
        return 0

    return value_of_tf_idf_in_doc


""" these two functions run the  program """

create_dictionary_for_documentNum_and_finalTokens_and_Count_function()
calculate_tf_idf_in_normalized_form_function()


""" Test of three functions: guery(qstring), getidf(token), and getweight(filename, token) """


"""
##################################################
print(" Testing of getidf function")

print("%.12f" % getidf("hispanic"))

print("%.12f" % getidf("agenda"))

print("%.12f" % getidf("health"))

print("++++++++++++++++++++++++++++")

####################################################
print("Testing of getweight function ")

print("%.12f" % getweight("2012-10-03.txt","health"))

print("%.12f" % getweight("1960-10-21.txt","reason"))

print("%.12f" % getweight("1976-10-22.txt","agenda"))


print("%.12f" % getweight("2012-10-16.txt","hispan"))

print("%.12f" % getweight("2012-10-16.txt","hispanic"))

print("++++++++++++++++++++++++++++")

#######################################################

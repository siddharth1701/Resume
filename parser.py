# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:37:34 2019

@author: zamirahmad.s
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:17:24 2019

@author: zamirahmad.s
"""

#importing Libraries
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from nltk.corpus import stopwords
import spacy
from spacy.matcher import Matcher
import re
import pandas as pd

# Extracting text from PDF:
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()

# calling above function and extracting text
file_path = "zamirCv4.pdf"
text = ''
for page in extract_text_from_pdf(file_path):
    text += ' ' + page
text = text.replace('(', ' ')
text = text.replace(')', '')   
# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

# Extract Name
def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
    matcher.add('NAME', None, pattern)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
    
# Extract Mobile Number
#def extract_mobile_number(text):
#    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9][2-9]{2})\s*(?:[.-]\s*)?([0-9]{6})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
#    
#    if phone:
#        number = ''.join(phone[0])
#        if len(number) > 10:
#            return '+' + number
#        else:
#            return number
        
def extract_mobile_numbers(text):
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9][2-9]{2})\s*(?:[.-]\s*)?([0-9]{6})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
    phones = []
    for number in phone:
        if phone:
            number = ''.join(phone[0])
            if len(number) > 10:
                num = '+' + number
                phones.append(num)
            else:
                phones.append(number)
    return phones
# Extract Email:
#def extract_email(email):
#    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
#    if email:
#        try:
#            return email[0].split()[0].strip(';')
#        except IndexError:
#            return None
        
def extract_emails(text):
    emails = []
    email = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    for mail in email:
        emails.append(mail)
    return emails

#    print(mail)
#Extract Education
# Grad all general stop words
#STOPWORDS = set(stopwords.words('english'))
## Education Degrees
#EDUCATION = [
#            'BE','B.E.', 'B.E', 'BS', 'B.S', 'B.SC.', 'BSC',
#            'M.E', 'M.E.', 'MS', 'M.S', 'M.SC.', 'MSC', 
#            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'MCA', 'M.C.A.', 'MBA', 'M.B.A'
#            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII', 'INTERMEDIATE'
#            ]
#def extract_education(resume_text):
#    nlp_text = nlp(resume_text)
#
#    # Sentence Tokenizer
#    nlp_text = [sent.string.strip() for sent in nlp_text.sents]
#
#    edu = {}
#    # Extract education degree
#    for index, text in enumerate(nlp_text):
#        for tex in text.split():
#            # Replace all special symbols
#            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
#            if tex.upper() in EDUCATION and tex not in STOPWORDS:
#                edu[tex] = text + nlp_text[index + 1]
#
#    # Extract year
#    education = []
#    for key in edu.keys():
#        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
#        if year:
#            education.append((key, ''.join(year[0])))
#        else:
#            education.append(key)
#    return education




def extract_education(resume_text):
    STOPWORDS = set(stopwords.words('english'))
# Education Degrees
    data = pd.read_csv("education.csv")    
    # extract values
    EDUCATION = list(data.columns.values)
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.lower() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


# load pre-trained model
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
noun_chunk = []
for chunk in doc.noun_chunks:
    # iterate over the noun chunks in the Doc
    noun_chunk.append(chunk.text)



def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # reading the csv file
    data = pd.read_csv("skills.csv") 
    
    # extract values
    skills = list(data.columns.values)
    
    skillset = []
    
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunk:
        token = token.lower().strip()
        if token in skills:
            skillset.append(token)
    
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

# Domain Extractor
def extract_domains(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # reading the csv file
    data = pd.read_csv("domains.csv") 
    
    # extract values
    domains = list(data.columns.values)
    
    domainset = []
    
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in domains:
            domainset.append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunk:
        token = token.lower().strip()
        if token in domains:
            domainset.append(token)
    
    return [i.capitalize() for i in set([i.lower() for i in domainset])]

name = extract_name(text)
phone = extract_mobile_numbers(text)
email = extract_emails(text)
education = extract_education(text)
skills = extract_skills(text)
domains = extract_domains(text)
from texttable import Texttable
t = Texttable()
t.set_cols_dtype(['t','t','t', 't', 't', 't']) 
t.set_cols_width([10, 16, 16, 10, 15, 15])
t.add_rows([['Name', 'Mobile Number', 'Email', 'Education', 'Technical Skills', 'Domains'], [name, phone, email, education, skills, domains]])
print(t.draw())
#print(tabulate([[name, phone, email, education]], headers=['Name', 'Mobile Number', 'Email', 'Education'], tablefmt='orgtbl'))



#import spacy


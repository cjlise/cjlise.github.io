---
title: "Web Scrapping Job Postings with Python and BeautifulSoup"
date: 2019-11-02
categories: software-engineering
tags: [Software Engineering, Python, Web Scrapping, BeautifulSoup, CSV]
header: 
   image: "/images/DeepLearning/web-3706561_200.jpg"
excerpt: "Software Engineering, Python, Web Scrapping, BeautifulSoup, CSV"
---

# Web Scrapping Job Postings with Python and BeautifulSoup
The goal of this project is to build a script to retrieve a data scientist job posting list from the Web site Indeed.fr. I've been inspired by this [Medium post](https://medium.com/@msalmon00/web-scraping-job-postings-from-indeed-96bd588dcb4b). The Medium post covers  Web scrapping from Indeed.com the US version of Indeed. Indeed.fr's design is a bit different, and I had to change and adapt the parsing/matching part. However I use the same kind of framework: 
* requests: To retrieve the job posting list
* BeautifulSoup: to parse and filter the job postings 
* Pandas: To store the job posting in a dataframe then write a csv file  

Currently the code collect about 100 job posting for the 3 regions Île-de-France,Nouvelle-Aquitaine,Provence-Alpes-Côte d'Azur. But this list can be updated. 

The CSV file created is in unicode utf-8 format. Here are the instructions to load it properly in Excel: 
* On the Excel Ribbon Select Data -> Get Data -> From File -> CSV File   
![get-data](/images/SoftwareEngineering/WebScrappingXL01.jpg "get-data from CSV")

* Select the option 650001: Unicode (UTF-8) then click on the load button 
![unicode](/images/SoftwareEngineering/WebScrappingXLUnicode.jpg "Select Unicode")

* Then Excel loads properly the specific French characters
![Excel](/images/SoftwareEngineering/WebScrappingXL02.jpg "File loaded in Excel")


The full Python code is listed below: 
```Python
# JobWebScrapping script
# Script to retrieve the list of data scientist job posting from indeed.fr 
# Scrap from indeed.fr the data scientist job for the list of region provided 
# And write the results in a csv file 
#
#  Author: José Lise 
#  November 2019
#
# Import the packages that we will use: Mainly Pandas, BeautifulSoup, requests
import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time


########################################################################
# Number maximum of results by region : This is an indicative parameter
# As we don't know exactly how many job posting is returned by a request
#
max_results_per_region = 100
# Region list: This can be changed according your wish
region_set = ['Île-de-France','Nouvelle-Aquitaine','Provence-Alpes-Côte d\'Azur']
# Columns of the CSV file 
columns = ["region", "job_title", "company_name", "location", "summary", "salary", "Date"]
sample_df = pd.DataFrame(columns = columns)

for region in region_set:
    start = 0
    n_region = 0
    while start < max_results_per_region :
        # preprare the query parameters
        payload = {'q':'data scientist','l':region,'start':str(start)}
        payload_str = "&".join("%s=%s" % (k,v) for k,v in payload.items())
        # Get the page with the results
        page = requests.get('http://www.indeed.fr/jobs', params=payload_str)
        time.sleep(1)  #ensuring at least 1 second between page grabs
        # Parse the page using lxml parser
        soup = BeautifulSoup(page.text, "lxml")
        #print(page.text)
        print(f"Region: {region} - start : {start} ")
        for div in soup.find_all(name="div", attrs={"class":"jobsearch-SerpJobCard unifiedRow row result"}): 
            #specifying row num for index of job posting in dataframe
            num = (len(sample_df) + 1)
            n_region += 1
            #creating an empty list to hold the data for each posting
            job_post = [] 
            #append region name
            job_post.append(region) 
            #grabbing job title
            for a in div.find_all(name="a", attrs={"data-tn-element":"jobTitle"}):
                job_post.append(a["title"])

            #grabbing company name
            company = div.find_all(name="span", attrs={"class":"company"}) 
            if len(company) > 0: 
                for b in company:
                   job_post.append(b.text.strip()) 
            else: 
                sec_try = div.find_all(name="span", attrs={"class":"result-link-source"})
                if len (sec_try) > 0:
                    for span in sec_try:
                       job_post.append(span.text)
                else:
                    job_post.append("Nothing_found")
            #grabbing location name
            c = div.findAll('span', attrs={'class': 'location'}) 
            if len(c) == 0 : 
                c = div.findAll('div', attrs={'class': 'location'})
            for span in c: 
                job_post.append(span.text) 
         
            #grabbing summary text
            d = div.findAll('div', attrs={'class': 'summary'}) 
            for span in d:
                job_post.append(span.text.strip()) 
            #grabbing salary
            try:
                div_two = div.find(name="div", attrs={"class":"salarySnippet salarySnippetDemphasize"}) 
                div_three = div_two.find("span", attrs={"class":"salary no-wrap"}) 
                job_post.append(div_three.text.strip())
            except:
                job_post.append("Nothing_found") 

            e = div.findAll('div', attrs={'class': 'jobsearch-SerpJobCard-footer'}) 
            try:
                div_two = div.find(name="div", attrs={"class":"jobsearch-SerpJobCard-footerActions"})
                div_three = div_two.find(name="div", attrs={"class":"result-link-bar-container"})
                div_four = div_three.find(name="span", attrs={"class":"date"})
                job_post.append(div_four.text.strip())
            except:
                job_post.append("Nothing_found") 

            #appending list of job post info to dataframe at index num
            sample_df.loc[num] = job_post
            print ( "Job posting ID: ", num)
        start=n_region
#saving sample_df as a local csv file — define your own local path to save contents 
sample_df.to_csv("JobPostingScrap.csv", encoding='utf-8')

```






import openreview #library for using the openReview API
import requests #library for using request function for the data that will flow through the API
import csv #library to import our output to a csv

import os #libraries needed for our file to recognize our .env keys
from dotenv import load_dotenv

load_dotenv() #loading our local .env file

#This is the data scrapping API which we need to get the data of the NeurIPS conference
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username=os.getenv('OPENREVIEW_USERNAME'),
    password=os.getenv('OPENREVIEW_KEY')
)

api2_venueid = 'NeurIPS.cc/2025/Conference' #critical info: if a paper has this venueid that means the paper has been accepted to the conference
#In 2024, NeurIPS used API 2
#api2_venue_group = live_client_v2.get_group(api2_venueid)
#api2_venue_domain = api2_venue_group.domain
#print(api2_venue_domain) 
#Output: NeurIPS.cc/2024/Conference

#get_all_notes is a built function of the OpenReview of REST API
#where numerous details of papers are included, titles, abstracts, authors etc...
#in our case we get the content of only venueid = Neurips2025 conference
submissions = client.get_all_notes(content={'venueid': api2_venueid})


#built around the logic of code retreived from docs.openreview.net
#we go through every item inside of submission, also we previously stated that it will be the items with venueid of only the Neurips2025 conference
#creating a csv file around the main code where we select the information we need
with open('neurips_accepted_submissions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Number', 'Title', 'Abstract', 'Authors'])  # header row

    # Loop inside the 'with' block
    for submission in submissions:
        title = submission.content.get('title', {}).get('value')
        abstract = submission.content.get('abstract', {}).get('value')
        author_ids = submission.content.get('authorids', {}).get('value', [])
        author_profiles = openreview.tools.get_profiles(client, author_ids)

        # Collect author names
        authors = []
        for author in author_profiles:
            authors.append(author.get_preferred_name(pretty=True))

        # Write submission to CSV
        writer.writerow([submission.number, title, abstract, "; ".join(authors)])






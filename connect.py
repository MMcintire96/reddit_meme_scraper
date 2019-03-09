import praw
import ast


f = open('creds.txt', 'r')
lines = f.readlines()
clean_lines = []
for line in lines:
    clean_lines.append(line.strip('\n'))

CLIENT_ID = clean_lines[0]
CLIENT_SECRET = clean_lines[1]
PASSWORD = clean_lines[2]
USERNAME = clean_lines[3]
USER_AGENT = clean_lines[4]

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                    password=PASSWORD, user_agent=USER_AGENT,
                    username=USERNAME)


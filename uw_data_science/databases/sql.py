######################################
# sql.py 
#
# Made to run sqlite3 in python to examine database files 
# 
# By David Curry
######################################

'''
import sqlite3
conn = sqlite3.connect('reuters.db')

c = conn.cursor()

c.execute(
' SELECT count(*) \
  FROM frequency  \
  WHERE docid="10398_txt_earn" '
) 

print c.fetchone()
'''

import pandas as pd
import sqlite3

conn = sqlite3.connect('reuters.db')

df = pd.read_sql('Select * From frequency', conn)

print df.head()

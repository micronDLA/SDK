#!/usr/bin python3

# E. Culurciello, February 2019
# add data to a database

# https://docs.python.org/2/library/sqlite3.html
# https://www.w3schools.com/sql/default.asp

import sqlite3
import datetime
import os.path


table_name = 'processed_images.db'
if not os.path.isfile(table_name):
	conn = sqlite3.connect(table_name)
	c = conn.cursor()
	# Create table
	c.execute('''CREATE TABLE processed_images
	         (date text, image text, data text)''')


def save_to_db(file, data):
	# open db:
	conn = sqlite3.connect(table_name)
	c = conn.cursor()
	# Insert a row of data
	date = datetime.datetime.now()
	# convert data to a string:
	s = ''
	for a in data:
		s = s + '; '+ a
	c.execute("INSERT INTO processed_images VALUES (?,?,?)", (date, file, s))
	# Save (commit) the changes
	conn.commit()
	# close db:
	conn.close()

	return True


if __name__ == '__main__':

	# test saving some random data:
	data = ['black-and-tan coonhound, 13.582031', 'Rottweiler, 12.878906', 'bloodhound, 11.738281', 'Gordon setter, 11.292969', 'Doberman, 11.1328125']
	save_to_db('dog224.jpg', data)

	# open db:
	conn = sqlite3.connect(table_name)
	c = conn.cursor()

	# print current db:
	print('\n\nPrinting entire db:')
	for row in c.execute('SELECT * FROM processed_images ORDER BY date'):
		print(row)

	# example search db:
	t = ("%bloodhound%",)
	c.execute('SELECT * FROM processed_images WHERE data like ?', t)
	print('\n\nSearch results:')
	print(c.fetchall())

	# close db:
	conn.close()
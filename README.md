# SM Template Project

This Project helps you to get started with Code at Smart Mobility Lab.

Please keep the following in mind:
1. Never commit your credentials or secrets into Git.
1. Never commit your credentials or secrets into Git.
1. Never commit your credentials or secrets into Git.

So proceed as following
1. Copy src/config.template.py into src/config.py and enter your user credentials. 
1. src/config.py should already be ignored by Git.
1. install the packages from requirements.txt into a virtual environment of your choice
1. start src/example.ipynb and connect the virtual environment in your IDE
1. Now run the example code to run your fist query. Congrats!
1. `run_sql` results are per default cached. Input parameters (i.e. the SQL query text) are recorded and along with all results saved into a hidden folder .cache/ . If you would like to rerun the query because the underlying data has changed, delete that folder or set `run_sql(..., cache=False, ...)`.
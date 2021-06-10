
.PHONY:

run:
	pipenv run python app.py

train:
	pipenv run python trainer.py

preanalyze:
	pipenv run python preanalyzer.py
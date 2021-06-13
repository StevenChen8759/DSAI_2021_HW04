
.PHONY:

answer: train inference

inference:
	pipenv run python main_inf.py

train:
	pipenv run python trainer.py

preanalyze:
	pipenv run python preanalyzer.py
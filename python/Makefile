train-model:
	@python3 neural/train.py

debug-model:
	@python3 neural/main.py

install-deps:
	@pip3 install -r requirements.txt

run-server:
	@uvicorn backend.main:app --reload

.PHONY: setup run test serve eval clean

setup:
	source venv/bin/activate

run:
	python pipeline.py full

ingest:
	python pipeline.py ingest

extract:
	python pipeline.py extract

trends:
	python pipeline.py trends

serve:
	python pipeline.py serve

test:
	python pipeline.py test
	pytest tests/test_pipeline.py -v

eval:
	python eval_extraction.py

clean:
	rm -f data/skillgap.db
	rm -rf data/output/* data/eval/eval_results.json data/cache/*
	@echo "Cleaned. Run 'python pipeline.py full' to rebuild."

stats:
	@python -c "from pipeline import Database; import json; print(json.dumps(Database().stats(), indent=2))"

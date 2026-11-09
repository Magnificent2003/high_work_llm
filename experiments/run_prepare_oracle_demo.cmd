@echo off
cd /d "D:\多模态小论文"
echo [1/3] Checking Python...
python --version
echo [2/3] Installing the small demo dependencies...
python -m pip install -r llm\requirements.txt
echo [3/3] Generating the first four-source evidence files...
python experiments\prepare_evidence_from_yolo_labels.py --dataset-root "input_demo\originals\Work at Height Safety Datase(DST2189)" --split valid --outdir "experiments\runs\oracle_evidence_002" --limit 20 --min-classes 2
echo.
echo Finished. Open experiments\runs\oracle_evidence_002 to inspect the outputs.
pause

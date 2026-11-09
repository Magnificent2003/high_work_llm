@echo off
cd /d "D:\多模态小论文"
echo Running the raw-image baseline...
python experiments\raw_only_baseline.py --image-dir "input_demo\originals\Work at Height Safety Datase(DST2189)\images\valid" --outdir "experiments\runs\raw_baseline_001" --env-file "llm\.env" --limit 20
echo.
echo Finished. Open experiments\runs\raw_baseline_001\raw_only_results.jsonl
pause

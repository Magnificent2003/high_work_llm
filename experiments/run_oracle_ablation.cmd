@echo off
cd /d "D:\多模态小论文"
echo Running the four-source ablation...
python experiments\four_source_ablation_demo.py --manifest "experiments\runs\oracle_evidence_002\manifest.json" --outdir "experiments\runs\oracle_ablation_001" --env-file "llm\.env" --conditions leave_one_out --repeats 3
echo.
echo Finished. Open experiments\runs\oracle_ablation_001\summary.json
pause

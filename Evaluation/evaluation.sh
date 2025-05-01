python evaluation.py \
  --clean  ../results/clean_predictions_twoclass_0.3.csv\
  --poisoned ../results/poisoned_predictions_twoclass_0.3.csv \
  --confidence_output_path confidence_histogram.png \
  --confusion_output_path confusion_matrix.png \
  --mcnemar_output_path mc_nemar.txt
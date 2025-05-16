if [ "$#" -ne 2 ]; then
	echo "Usage: ${0} directory random_seed"
	exit 1
fi

DIR=$1
SEED=$2

python label_generation/get_best_hyperparameters_binary_classification.py --directory ${DIR} -s ${SEED}
python label_generation/generate_labels_binary_classification.py --directory ${DIR} -s ${SEED}

# Clean up
rm "label_generation/get_best_hyperparameters_binary_classification_s${SEED}.log"
rm "label_generation/generate_labels_binary_classification_roc_auc_s${SEED}.log"
for d in "${DIR}"/*; do
	if [ -d "${d}" ]; then
		cd "${d}"
		rm "best_params_s${SEED}.json"
		cd ..
	fi
done

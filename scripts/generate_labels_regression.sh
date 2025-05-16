if [ "$#" -ne 2 ]; then
	echo "Usage: ${0} directory random_seed"
	exit 1
fi

DIR=$1
SEED=$2

python label_generation/get_best_hyperparameters_regression.py --directory ${DIR} -s ${SEED}
python label_generation/generate_labels_regression.py --directory ${DIR} -s ${SEED}

# Clean up
rm "label_generation/get_best_hyperparameters_regression_s${SEED}.log"
rm "label_generation/generate_labels_regression_neg_mean_absolute_percentage_error_s${SEED}.log"
for d in "${DIR}"/*; do
	if [ -d "${d}" ]; then
		cd "${d}"
		rm "best_params_s${SEED}.json"
		cd ..
	fi
done

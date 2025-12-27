import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def parse_args() -> argparse.Namespace:
	'''
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	'''
	parser = argparse.ArgumentParser(description='Generate prompts for the TTI counting task.')
	parser.add_argument('--n_shapes', type=int, nargs='+', default=[2,3,4,5,6,7,8,9,10], help='Number of stimuli to present.')
	parser.add_argument('--objects', type=str, nargs='+', default=['sphere', 'cone', 'cube', 'pyramid'], help='Names of the shapes to use in the trials.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--output_dir', type=str, default='data/tti/counting', help='Directory to save the generated trials.')
	return parser.parse_args()


def main():
	# Fix the random seed for reproducibility.
	np.random.seed(88)
	
	# Parse command line arguments.
	args = parse_args()
	basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	
	# Create directory for serial search exists.
	os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

	# Initialize DataFrame for storing task metadata_df.
	metadata_df = pd.DataFrame(columns=['path', 'prompt', 'object', 'n_objects', 'completed'])

	# Generate the trials.
	for n in tqdm(args.n_shapes):
		for i in range(args.n_trials):
			# Save the trials and their metadata.
			object_name = np.random.choice(args.objects)
			prompt = f'Exactly {n} solid black {object_name} shaped objects on a white background with no other objects.'
			trial_path = os.path.join(basepath, args.output_dir, 'images', f'trial-{n}_{i}.png')
			trial_metadata = {'path': trial_path, 'prompt': prompt, 'object': object_name, 'n_shapes': n, 'completed': False}
			metadata_df = metadata_df._append(trial_metadata, ignore_index=True)

	# Save results DataFrame to CSV.
	metadata_df.to_csv(os.path.join(basepath, args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()
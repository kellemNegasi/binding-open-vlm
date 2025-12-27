import argparse
import os
import pandas as pd
from itertools import product
import numpy as np
from typing import List
from tqdm import tqdm
import sys

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def make_binding_trial(color_names: List[str],
					   shape_names: List[str], 
					   n_objects: int = 5, 
					   n_shapes: int = 5, 
					   n_colors: int = 5):
	# sample the shapes and colors of objects to include in the trial.
	unique_shape_inds = np.random.choice(len(shape_names), n_shapes, replace=False) # sample the unique shapes for the current trial.
	shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, n_objects-n_shapes, replace=True)])
	unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)  # sample the unique colors for the current trial.
	color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, n_objects-n_colors, replace=True)])
	colors = color_names[color_inds]
	shapes = shape_names[shape_inds]
	object_features = [color+' '+shape for shape, color in zip(shapes, colors)]
	return object_features

def parse_args() -> argparse.Namespace:
	'''
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	'''
	parser = argparse.ArgumentParser(description='Generate prompts for the TTI binding task.')
	parser.add_argument('--n_objects', type=int, nargs='+', default=[4], help='Number of stimuli to present.')
	parser.add_argument('--object_names', type=str, nargs='+', default=['sphere', 'cone', 'cube', 'cylinder', 'teacup'], help='Names of the shapes to use in the trials.')
	parser.add_argument('--color_names', type=str, nargs='+', default=['red', 'green', 'blue', 'yellow', 'black'], help='Names of the colors to use in the trials.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--output_dir', type=str, default='data/tti/binding', help='Directory to save the generated trials.')
	return parser.parse_args()


def main():
	# Fix the random seed for reproducibility.
	np.random.seed(88)
	
	# Parse command line arguments.
	args = parse_args()
	assert len(args.object_names) == len(args.color_names) 
	assert np.max(args.n_objects)<=len(args.object_names)
	assert np.max(args.n_objects)<=len(args.color_names)
	
	# Create directory for serial search exists.
	basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	os.makedirs(os.path.join(basepath, args.output_dir, 'images'), exist_ok=True)

	# Initialize DataFrame for storing task metadata_df.
	metadata_df = pd.DataFrame(columns=['path', 'prompt', 'n_objects', 'n_shapes', 'n_colors', 'features', 'completed'], dtype=object)

	# Generate the trials.
	for n in tqdm(args.n_objects):
		# Task conditions that we want to generate trials for.
		task_conditions = list(product(range(1,n+1), range(1,n+1)))
		condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
		counts, count_freq = np.unique(condition_feature_counts, return_counts=True)

		# Generate trials for each task condition.
		for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):

			# Find how many trials to generate for each task condition to ensure nTrials per nFeatures condition.
			n_trials = int(np.ceil(args.n_trials / count_freq[counts==n_features][0]))

			for i in range(n_trials):
				features = make_binding_trial(np.array(args.color_names), np.array(args.object_names), n_objects=n, n_shapes=n_shapes, n_colors=n_colors)
				objects, obj_counts = np.unique(features, return_counts=True)
				objects = '\n'.join([f'{c} {o}s' if c>1 else f'{c} {o}' for o, c in zip(objects, obj_counts)])
				prompt = f'An image with exactly {n} objects equally spaced such that each one is visible against a white background. The following objects are present in the image: \n{objects}'
				trial_path = os.path.join(basepath, args.output_dir, 'images', f'nObjects={n}_nShapes={n_shapes}_nColors={n_colors}_{i}.png')
				trial_metadata = {'path': trial_path, 
					  			  'prompt': prompt, 
								  'n_objects': n, 
								  'n_shapes': n_shapes, 
								  'n_colors': n_colors,
								  'features': features,
								  'completed': False}
				metadata_df = metadata_df._append(trial_metadata, ignore_index=True)

	# Save results DataFrame to CSV.
	metadata_df.to_csv(os.path.join(basepath, args.output_dir, 'metadata.csv'), index=False)

if __name__ == '__main__':
	main()
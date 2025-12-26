import argparse
from functools import reduce
import os
import json
from tqdm import tqdm

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys

# Add the parent directory to the sys.path list so we can import utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def make_rmts_trial(source_ind: int, feature_df: pd.DataFrame, pair_imgs: np.array, relation: str='color'):
	'''
	Make a single trial.

	Args:
		source_ind: Index of the source pair in the feature_df.
		feature_df: DataFrame of features, with columns 'shape1', 'color1', 'shape2', 'color2', 'same_color', 'same_shape', 'same_object'.
		pair_imgs: Array of images with shape (n_pairs, 2, n_channels, width, height).
		relation: The relation to use for the trial (color or shape).
	'''
	# Find all pairs that match the source pair in the given relation
	source_features = feature_df.iloc[feature_df.index==source_ind].reset_index(drop=True)
	source_imgs = pair_imgs[source_ind]
	source_relations = source_features[['same_shape', 'same_color']]
	inds = [pair_df[i]==j for i,j in zip(source_relations.columns, source_relations.values[0])] 
	correct_inds = reduce(lambda x, y: x & y, inds)
	correct_target_features = feature_df[correct_inds & (feature_df.index != source_ind)].sample(1)
	correct_target_imgs = pair_imgs[correct_target_features.index[0]].squeeze()
	# Find all pairs that differ from the source pair only in the given relation
	color_mask = feature_df.same_color==source_relations.same_color.values[0]
	shape_mask = feature_df.same_shape==source_relations.same_shape.values[0]
	if relation == 'color':
		incorrect_inds = ~color_mask & shape_mask
	elif relation == 'shape':
		incorrect_inds = color_mask & ~shape_mask
	else:
		raise ValueError('Relation must be "color" or "shape"')
	# Randomly select one pair as the incorrect target.
	incorrect_target_features = feature_df[incorrect_inds].sample(1)
	incorrect_target_imgs = pair_imgs[incorrect_target_features.index[0]].squeeze()
	# Randomly assign the side to present the correct target.
	correct_side = np.random.rand() > 0.5
	if correct_side:
		target1_features = correct_target_features.add_prefix('target1_').reset_index(drop=True)
		target2_features = incorrect_target_features.add_prefix('target2_').reset_index(drop=True)
		target1_imgs, target2_imgs = correct_target_imgs, incorrect_target_imgs
		correct_target = 1
	else:
		target1_features = incorrect_target_features.add_prefix('target1_').reset_index(drop=True)
		target2_features = correct_target_features.add_prefix('target2_').reset_index(drop=True)
		target1_imgs, target2_imgs = incorrect_target_imgs, correct_target_imgs
		correct_target = 2
	# Combine the features into a single dataframe, and return the trial images.
	source_features = source_features.add_prefix('source_').reset_index(drop=True)
	trial_features = pd.concat([source_features, target1_features, target2_features], axis=1)
	trial_features['correct_target'] = correct_target
	trial_features['task_relation'] = f'same_{relation}'
	return trial_features, source_imgs, target1_imgs, target2_imgs


def make_trial(source_pair, target_pair1, target_pair2):
	# Define the canvas to draw images on, font, and drawing tool.
	canvas_img = Image.new('RGB', (512, 512), (255, 255, 255))
	# Helper function to paste images and add labels
	def paste_images_and_label(image_pair, position, label):
		img1, img2 = image_pair[0], image_pair[1]
		img1 = resize(img1.astype(np.uint8), 96)
		img2 = resize(img2.astype(np.uint8), 96)
		img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
		img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))
		# Create a new image to hold the side-by-side pair
		pair_img = Image.new('RGB', (256, 256), (255, 255, 255))
		pair_img.paste(img1, (16, 96))
		pair_img.paste(img2, (128, 96))
		# Add the label below the pair image
		draw = ImageDraw.Draw(pair_img)
		font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Black.ttf', size=28)
		text_size = draw.textlength(label, font=font)
		label_position = (128 - text_size//2, 40)
		draw.text(label_position, label, (0, 0, 0), font=font)
		# Paste the pair image onto the canvas
		canvas_img.paste(pair_img, position)
		return pair_img
	# Add the image pairs to the canvas.
	source_img = paste_images_and_label(source_pair, (128, 0), 'Source Pair')
	target1_img = paste_images_and_label(target_pair1, (0, 256), 'Target Pair 1')
	target2_img = paste_images_and_label(target_pair2, (256, 256), 'Target Pair 2')
	# Convert the final canvas back to a NumPy array
	return canvas_img, source_img, target1_img, target2_img


def gen_all_shapes(imgs, rgb_values, colors):
	# Generate all possible shapes.
	all_features = []
	all_objects = []
	for i, img in enumerate(imgs):
		for j, rgb in enumerate(rgb_values):
			rgb_img = color_shape(img.astype(np.float32), rgb)
			all_features.append([args.shape_names[i], colors[j]])
			all_objects.append(rgb_img)
	object_df = pd.DataFrame(all_features, columns=['shape', 'color'])
	object_features = np.array(all_features)
	all_objects = np.stack(all_objects)
	return object_df, object_features, all_objects


def gen_all_pairs(object_features, all_objects):
	# Generate all possible pairs of shapes.
	feat1 = np.repeat(object_features, len(object_features), axis=0)
	feat2 = np.tile(object_features, (len(object_features), 1))
	shapes1 = np.repeat(all_objects, len(all_objects), axis=0)
	shapes2 = np.tile(all_objects, (len(all_objects),1,1,1))
	pair_features = np.hstack([feat1, feat2])
	pair_df = pd.DataFrame(pair_features, columns=['shape1', 'color1', 'shape2', 'color2'])
	pair_df['same_color'] = pair_df['color1'] == pair_df['color2']
	pair_df['same_shape'] = pair_df['shape1'] == pair_df['shape2']
	pair_df['same_object'] = (pair_df['same_color'].values) & (pair_df['same_shape'].values)
	# Filter out pairs that we don't want.
	good_inds = (pair_df['same_color'] | pair_df['same_shape'])
	pair_df = pair_df[good_inds].reset_index(drop=True)
	pair_shapes = np.stack([shapes1[good_inds], shapes2[good_inds]], axis=1) # stack the shapes into a single tensor.
	return pair_df, pair_shapes


def get_trial_json(row):
	pairs = ['source', 'target1', 'target2']
	objects = ['1', '2']
	trial = {}
	for pair in pairs:
		trial[pair] = {}
		for obj in objects:
			trial[pair][f'{pair}_object{obj}'] = {'shape': row[f'{pair}_shape{obj}'], 'color': row[f'{pair}_color{obj}']}

	# convert all of the dictionary to JSON strings
	return trial # json.dumps(trial)


def parse_args() -> argparse.Namespace:
	'''
	Parse command line arguments.

	Returns:
	argparse.Namespace: The parsed command line arguments.
	'''
	parser = argparse.ArgumentParser(description='Generate feature binding trials.')
	parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to generate per n_shapes condition.')
	parser.add_argument('--color_names', type=str, nargs='+', default=['red', 'green', 'blue', 'purple', 'saddlebrown', 'black'], help='Colors to use for the shapes.')
	parser.add_argument('--shape_names', type=str, nargs='+', default=['triangle', 'star', 'heart', 'cross', 'circle', 'square'], help='Names of the shapes to use in the trials.')
	parser.add_argument('--shape_inds', type=int, nargs='+', default=[9,98,96,24,100,101], help='Indices of the shapes to include in the trials.')
	parser.add_argument('--output_dir', type=str, default='data/vlm/rmts', help='Directory to save the generated trials.')
	return parser.parse_args()


if __name__=='__main__':
	# Fix the random seed for reproducibility.
	np.random.seed(88)

	# Load the base objects.
	args = parse_args()
	basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	imgs = np.load(os.path.join('imgs.npy'))
	imgs = imgs[args.shape_inds]
	rgb_values = np.array([mcolors.to_rgb(color) for color in args.color_names])
	assert len(args.shape_names) == len(args.shape_inds) 
	assert len(args.color_names) == len(args.shape_names) 

	# Create directory for RMTS task.
	os.makedirs(os.path.join(basepath, args.output_dir, 'images'), exist_ok=True)

	# Generate all possible shapes.
	object_df, object_features, all_objects = gen_all_shapes(imgs, rgb_values, args.color_names)

	# Generate all possible pairs of shapes.
	pair_df, pair_shapes = gen_all_pairs(object_features, all_objects)

	# Metadata for the trials.
	trial_metadata = []
	features = ['color', 'shape',]
	pairs = [('source', 'top'), ('target1', 'bottom left'), ('target2', 'bottom right')]
	objects = [(1, '1st', 'left'), (2, '2nd', 'right')]
	feature_task_columns = ['unified_path', 'decomposed_paths', 'feature', 'feature_value', 'pair', 'pair_loc', 'object_loc', 'object_ind']
	relation_task_columns = ['unified_path', 'decomposed_paths', 'relation', 'relation_value', 'pair', 'pair_loc']
	full_task_columns = ['unified_path', 'decomposed_paths', 'correct']
	feature_task_df = pd.DataFrame(np.zeros([args.n_trials, len(feature_task_columns)]), columns=feature_task_columns, dtype=object)
	relation_task_df = pd.DataFrame(np.zeros([args.n_trials, len(relation_task_columns)]), columns=relation_task_columns, dtype=object)
	full_task_df = pd.DataFrame(np.zeros([args.n_trials, len(full_task_columns)]), columns=full_task_columns, dtype=object)

	# Generate the trials.
	for n in tqdm(range(args.n_trials)):

		# normalize the source pair probabilities based on whether the pair is identical or not.
		value_counts = pair_df.same_object.value_counts()
		probs = value_counts / value_counts.sum()
		source_probs = np.zeros(len(pair_df))
		source_probs[pair_df.same_object==True] = 0.5 / probs[True]
		source_probs[pair_df.same_object==False] = 0.5 / probs[False]
		source_probs = source_probs / source_probs.sum()

		# sample a random pair index for the source.
		source_ind = np.random.choice(len(pair_df), p=source_probs)
		feature_inds = np.where(pair_df.loc[source_ind, ['same_shape', 'same_color']].values)[0] # valid relations only
		feature = features[np.random.choice(feature_inds)]
		trial_features, source_pair, target_pair1, target_pair2 = make_rmts_trial(source_ind, pair_df, pair_shapes, relation=feature)
		canvas_img, source_img, target1_img, target2_img = make_trial(source_pair, target_pair1, target_pair2)
		trial_metadata.append(trial_features)

		# Save the task images.
		unified_path = os.path.join(basepath, args.output_dir, 'images', f'trial-{n}.png')
		source_path = os.path.join(basepath, args.output_dir, 'images', f'source-{n}.png')
		target1_path = os.path.join(basepath, args.output_dir, 'images', f'target1-{n}.png')
		target2_path = os.path.join(basepath, args.output_dir, 'images', f'target2-{n}.png')
		decomposed_paths = [source_path, target1_path, target2_path]
		canvas_img.save(unified_path)
		source_img.save(source_path)
		target1_img.save(target1_path)
		target2_img.save(target2_path)

		# Save the trial metadata.
		feature = features[np.random.choice(len(features))]
		pair, pair_loc = pairs[np.random.choice(len(pairs))]
		obj_id, obj_index, obj_loc = objects[np.random.choice(len(objects))]
		full_task_df.loc[n] = [unified_path, str(decomposed_paths), trial_features.correct_target[0]]
		feature_value = trial_features[f'{pair}_{feature}{obj_id}'].values[0]
		relation_value = trial_features[f'{pair}_same_{feature}'].values[0]
		feature_task_df.loc[n] = [unified_path, str(decomposed_paths), feature, feature_value, pair, pair_loc, obj_loc, obj_index]
		relation_task_df.loc[n] = [unified_path, str(decomposed_paths), f'same_{feature}', relation_value, pair, pair_loc]

	# Save the trial metadata.
	trial_metadata_df = pd.concat(trial_metadata, axis=0)
	trial_metadata_df.to_csv(os.path.join(basepath, args.output_dir, 'trial_metadata.csv'), index=False)
	feature_task_df.to_csv(os.path.join(basepath, args.output_dir, 'feature_task_metadata.csv'), index=False)
	relation_task_df.to_csv(os.path.join(basepath, args.output_dir, 'relation_task_metadata.csv'), index=False)
	full_task_df.to_csv(os.path.join(basepath, args.output_dir, 'full_task_metadata.csv'), index=False)

	trial = trial_metadata_df.apply(get_trial_json, axis=1).reset_index()
	feature_task2_df = full_task_df[['unified_path', 'decomposed_paths']]
	feature_task2_df['features'] = trial[0]
	feature_task2_df.to_csv(os.path.join(basepath, args.output_dir, 'feature_task2_metadata.csv'), index=False)
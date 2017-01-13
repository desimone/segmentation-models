"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import, division, print_function

from datasets import pascal

datasets_map = {'pascal': pascal}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset. """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(split_name, dataset_dir, file_pattern,
                                      reader)

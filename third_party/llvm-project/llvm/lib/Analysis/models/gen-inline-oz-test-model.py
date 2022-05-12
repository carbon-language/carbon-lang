"""Generate a mock model for LLVM tests.

The generated model is not a neural net - it is just a tf.function with the
correct input and output parameters. By construction, the mock model will always
output 1.
"""

import os
import importlib.util
import sys

import tensorflow as tf

POLICY_DECISION_LABEL = 'inlining_decision'
POLICY_OUTPUT_SPEC = """
[
    {
        "logging_name": "inlining_decision",
        "tensor_spec": {
            "name": "StatefulPartitionedCall",
            "port": 0,
            "type": "int64_t",
            "shape": [
                1
            ]
        }
    }
]
"""


# pylint: disable=g-complex-comprehension
def get_input_signature():
  """Returns the list of features for LLVM inlining."""
  # int64 features
  inputs = [
      tf.TensorSpec(dtype=tf.int64, shape=(), name=key) for key in [
          'caller_basic_block_count',
          'caller_conditionally_executed_blocks',
          'caller_users',
          'callee_basic_block_count',
          'callee_conditionally_executed_blocks',
          'callee_users',
          'nr_ctant_params',
          'node_count',
          'edge_count',
          'callsite_height',
          'cost_estimate',
          'inlining_default',
          'sroa_savings',
          'sroa_losses',
          'load_elimination',
          'call_penalty',
          'call_argument_setup',
          'load_relative_intrinsic',
          'lowered_call_arg_setup',
          'indirect_call_penalty',
          'jump_table_penalty',
          'case_cluster_penalty',
          'switch_penalty',
          'unsimplified_common_instructions',
          'num_loops',
          'dead_blocks',
          'simplified_instructions',
          'constant_args',
          'constant_offset_ptr_args',
          'callsite_cost',
          'cold_cc_penalty',
          'last_call_to_static_bonus',
          'is_multiple_blocks',
          'nested_inlines',
          'nested_inline_cost_estimate',
          'threshold',
      ]
  ]

  # float32 features
  inputs.extend([
      tf.TensorSpec(dtype=tf.float32, shape=(), name=key)
      for key in ['discount', 'reward']
  ])

  # int32 features
  inputs.extend([
      tf.TensorSpec(dtype=tf.int32, shape=(), name=key)
      for key in ['step_type']
  ])
  return inputs


def get_output_signature():
  return POLICY_DECISION_LABEL


def get_output_spec():
  return POLICY_OUTPUT_SPEC

def get_output_spec_path(path):
  return os.path.join(path, 'output_spec.json')


def build_mock_model(path, signature):
  """Build and save the mock model with the given signature"""
  module = tf.Module()

  # We have to set this useless variable in order for the TF C API to correctly
  # intake it
  module.var = tf.Variable(0.)

  def action(*inputs):
    s = tf.reduce_sum([tf.cast(x, tf.float32) for x in tf.nest.flatten(inputs)])
    return {signature['output']: float('inf') + s + module.var}

  module.action = tf.function()(action)
  action = {'action': module.action.get_concrete_function(signature['inputs'])}
  tf.saved_model.save(module, path, signatures=action)

  output_spec_path = get_output_spec_path(path)
  with open(output_spec_path, 'w') as f:
    print(f'Writing output spec to {output_spec_path}.')
    f.write(signature['output_spec'])


def get_signature():
  return {
      'inputs': get_input_signature(),
      'output': get_output_signature(),
      'output_spec': get_output_spec()
  }


def main(argv):
  assert len(argv) == 2
  model_path = argv[1]

  print(f'Output model to: [{argv[1]}]')
  signature = get_signature()
  build_mock_model(model_path, signature)


if __name__ == '__main__':
  main(sys.argv)

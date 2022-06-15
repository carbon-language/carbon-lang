# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl


def run(f):
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      print("\nTEST:", f.__name__)
      f()
    print(module)
  return f


@run
def testSequenceOp():
  sequence = transform.SequenceOp([pdl.OperationType.get()])
  with InsertionPoint(sequence.body):
    transform.YieldOp([sequence.bodyTarget])
  # CHECK-LABEL: TEST: testSequenceOp
  # CHECK: = transform.sequence {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   yield %[[ARG0]] : !pdl.operation
  # CHECK: } : !pdl.operation


@run
def testNestedSequenceOp():
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    nested = transform.SequenceOp(sequence.bodyTarget)
    with InsertionPoint(nested.body):
      doubly_nested = transform.SequenceOp([pdl.OperationType.get()],
                                           nested.bodyTarget)
      with InsertionPoint(doubly_nested.body):
        transform.YieldOp([doubly_nested.bodyTarget])
      transform.YieldOp()
    transform.YieldOp()
  # CHECK-LABEL: TEST: testNestedSequenceOp
  # CHECK: transform.sequence {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   sequence %[[ARG0]] {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:     = sequence %[[ARG1]] {
  # CHECK:     ^{{.*}}(%[[ARG2:.+]]: !pdl.operation):
  # CHECK:       yield %[[ARG2]] : !pdl.operation
  # CHECK:     } : !pdl.operation
  # CHECK:   }
  # CHECK: }


@run
def testTransformPDLOps():
  withPdl = transform.WithPDLPatternsOp()
  with InsertionPoint(withPdl.body):
    sequence = transform.SequenceOp([pdl.OperationType.get()],
                                    withPdl.bodyTarget)
    with InsertionPoint(sequence.body):
      match = transform.PDLMatchOp(sequence.bodyTarget, "pdl_matcher")
      transform.YieldOp(match)
  # CHECK-LABEL: TEST: testTransformPDLOps
  # CHECK: transform.with_pdl_patterns {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !pdl.operation):
  # CHECK:   = sequence %[[ARG0]] {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:     %[[RES:.+]] = pdl_match @pdl_matcher in %[[ARG1]]
  # CHECK:     yield %[[RES]] : !pdl.operation
  # CHECK:   } : !pdl.operation
  # CHECK: }


@run
def testGetClosestIsolatedParentOp():
  sequence = transform.SequenceOp()
  with InsertionPoint(sequence.body):
    transform.GetClosestIsolatedParentOp(sequence.bodyTarget)
    transform.YieldOp()
  # CHECK-LABEL: TEST: testGetClosestIsolatedParentOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !pdl.operation):
  # CHECK:   = get_closest_isolated_parent %[[ARG1]]

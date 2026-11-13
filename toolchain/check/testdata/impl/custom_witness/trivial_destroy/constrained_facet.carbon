// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon

// --- todo_constrained_facet.carbon
library "[[@TEST_NAME]]";

fn Test(generic T: Core.TrivialDestroy) {
  typeof(T) as Core.SubobjectDestroy;
  typeof(T) as Core.Destroy;
  typeof(T) as Core.TrivialDestroy;
}

// --- todo_generic_class.carbon
library "[[@TEST_NAME]]";

class GenericClass(T: Core.TrivialDestroy) {
  var x: T;
}

fn Test(generic T: Core.TrivialDestroy) {
  typeof(GenericClass(T)) as Core.SubobjectDestroy;
  typeof(GenericClass(T)) as Core.Destroy;
  typeof(GenericClass(T)) as Core.TrivialDestroy;
}

// --- todo_struct.carbon
library "[[@TEST_NAME]]";

fn Test(generic T: Core.TrivialDestroy) {
  alias Struct = {.a: T};

  typeof(Struct) as Core.SubobjectDestroy;
  typeof(Struct) as Core.Destroy;
  typeof(Struct) as Core.TrivialDestroy;
}

// --- todo_tuple.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

fn Test(generic T: Core.TrivialDestroy) {
  alias Tuple = (T, HasTrivialDestruction);

  typeof(Tuple) as Core.SubobjectDestroy;
  typeof(Tuple) as Core.Destroy;
  typeof(Tuple) as Core.TrivialDestroy;
}


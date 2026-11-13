// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon

// --- todo_constrained_facet.carbon
library "[[@TEST_NAME]]";

fn Make(generic T: Core.TrivialDestroy) -> typeof(T);

fn Test(generic T: Core.TrivialDestroy) {
  typeof(T) as Core.SubobjectDestroy;
  typeof(T) as Core.Destroy;
  typeof(T) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(T) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_generic_class.carbon
library "[[@TEST_NAME]]";

class GenericClass(T: Core.TrivialDestroy) {
  var x: T;
}

fn Make(generic T: Core.TrivialDestroy) -> GenericClass(T);

fn Test(generic T: Core.TrivialDestroy) {
  typeof(GenericClass(T)) as Core.SubobjectDestroy;
  typeof(GenericClass(T)) as Core.Destroy;
  typeof(GenericClass(T)) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(GenericClass(T)) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_struct.carbon
library "[[@TEST_NAME]]";

fn Make(generic T: Core.TrivialDestroy) -> {.a: T};

fn Test(generic T: Core.TrivialDestroy) {
  alias Struct = {.a: T};

  typeof(Struct) as Core.SubobjectDestroy;
  typeof(Struct) as Core.Destroy;
  typeof(Struct) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(Struct) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_tuple.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

fn Make(generic T: Core.TrivialDestroy) -> (T, HasNonTrivialDestruction);

fn Test(generic T: Core.TrivialDestroy) {
  alias Tuple = (T, HasTrivialDestruction);

  typeof(Tuple) as Core.SubobjectDestroy;
  typeof(Tuple) as Core.Destroy;
  typeof(Tuple) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(Tuple) = Make(T);
  //@dump-sem-ir-end
}

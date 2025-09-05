// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_final.carbon

// --- extend_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

class C {
  extend final impl as Z where .X = () {}
}

fn F() {
  let b: C.X = ();
}

// --- fail_final_extend_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  // CHECK:STDERR: fail_final_extend_impl.carbon:[[@LINE+7]]:9: error: `extend` must appear before `final` [ModifierMustAppearBefore]
  // CHECK:STDERR:   final extend impl as Z {}
  // CHECK:STDERR:         ^~~~~~
  // CHECK:STDERR: fail_final_extend_impl.carbon:[[@LINE+4]]:3: note: `final` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   final extend impl as Z {}
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final extend impl as Z {}
}

// --- fail_final_extend_impl_outside_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {}

// CHECK:STDERR: fail_final_extend_impl_outside_class.carbon:[[@LINE+4]]:1: error: `extend impl` can only be used in a class [ExtendImplOutsideClass]
// CHECK:STDERR: extend final impl C as Z {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extend final impl C as Z {}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/size_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/size_mismatch.carbon

// --- fail_assign_from_empty.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_from_empty.carbon:[[@LINE+4]]:16: error: cannot initialize tuple of 1 element from tuple with 0 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((),) = ();
// CHECK:STDERR:                ^~
// CHECK:STDERR:
var x: ((),) = ();

// --- fail_assign_to_empty.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+4]]:13: error: cannot initialize tuple of 0 elements from tuple with 1 element [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: () = ((),);
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
var x: () = ((),);

// --- fail_assign_nested.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+4]]:32: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));
// CHECK:STDERR:                                ^~~~~~~~~~~~
// CHECK:STDERR:
var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));

// --- fail_too_few_elements.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few_elements.carbon:[[@LINE+4]]:19: error: cannot initialize tuple of 2 elements from tuple with 1 element [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((), ()) = ((), );
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var x: ((), ()) = ((), );

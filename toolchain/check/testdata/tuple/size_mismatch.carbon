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

// CHECK:STDERR:        ╭─┤ fail_assign_from_empty.carbon:[[@LINE+8]]:16
// CHECK:STDERR: ->     │ var x: ((),) = ();
// CHECK:STDERR:        ·                ┳━
// CHECK:STDERR:        ·                ┗━━┫ error: cannot initialize tuple of 1 element from tuple with
// CHECK:STDERR:        ·                   ┃        0 elements
// CHECK:STDERR:        ·                   ┃        [TupleInitElementCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: ((),) = ();

// --- fail_assign_to_empty.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_assign_to_empty.carbon:[[@LINE+8]]:13
// CHECK:STDERR: ->     │ var x: () = ((),);
// CHECK:STDERR:        ·             ┳━━━━
// CHECK:STDERR:        ·             ┗━━┫ error: cannot initialize tuple of 0 elements from tuple with 1
// CHECK:STDERR:        ·                ┃        element
// CHECK:STDERR:        ·                ┃        [TupleInitElementCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: () = ((),);

// --- fail_assign_nested.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_assign_nested.carbon:[[@LINE+8]]:32
// CHECK:STDERR: ->     │ var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));
// CHECK:STDERR:        ·                                ┳━━━━━━━━━━━
// CHECK:STDERR:        ·                                ┗━━┫ error: cannot initialize tuple of 2 elements
// CHECK:STDERR:        ·                                   ┃        from tuple with 3 elements
// CHECK:STDERR:        ·                                   ┃        [TupleInitElementCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));

// --- fail_too_few_elements.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_too_few_elements.carbon:[[@LINE+8]]:19
// CHECK:STDERR: ->     │ var x: ((), ()) = ((), );
// CHECK:STDERR:        ·                   ┳━━━━━
// CHECK:STDERR:        ·                   ┗━━┫ error: cannot initialize tuple of 2 elements from tuple
// CHECK:STDERR:        ·                      ┃        with 1 element
// CHECK:STDERR:        ·                      ┃        [TupleInitElementCountMismatch]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: ((), ()) = ((), );

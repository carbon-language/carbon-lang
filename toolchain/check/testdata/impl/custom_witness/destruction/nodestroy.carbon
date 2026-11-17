// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nodestroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destruction/nodestroy.carbon

// --- fail_nodestroy_incomplete_type_subobject_destroy.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_incomplete_type_subobject_destroy.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Incomplete as Core.SubobjectDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Incomplete` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.SubobjectDestroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Incomplete as Core.SubobjectDestroy;
}

// --- fail_nodestroy_incomplete_type_destroy.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_incomplete_type_destroy.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Incomplete as Core.Destroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Incomplete` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Incomplete as Core.Destroy;
}

// --- fail_nodestroy_incomplete_type_trivial_destroy.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_nodestroy_incomplete_type_trivial_destroy.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Incomplete as Core.TrivialDestroy;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Incomplete` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Destroy`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Incomplete as Core.TrivialDestroy;
}

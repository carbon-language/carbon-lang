// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/comp_time_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/comp_time_field.carbon

// --- fail_let.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_let.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let A:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let A:! type = Class;

  // CHECK:STDERR: fail_let.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let template B:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let template B:! type = Class;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_var.carbon:[[@LINE+8]]:8: error: expected `:` in field declaration [ExpectedFieldColon]
  // CHECK:STDERR:   var C:! type = Class;
  // CHECK:STDERR:        ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var C:! type = Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var C:! type = Class;

  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var template D:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  var template D:! type = Class;
}

var x: Class = {};

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/scope.carbon

// --- scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> () {
    return ();
  }

  fn G() -> () {
    return F();
  }
}

fn F() -> () {
  return ();
}

fn Run() {
  var a: () = F();
  var b: () = Class.F();
}

// --- reenter_scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> ();
  fn G() -> ();
}

fn Class.F() -> () {
  Self.G();
  return G();
}

// --- fail_out_of_scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> () {
    return ();
  }
}

fn G() -> () {
  // CHECK:STDERR: fail_out_of_scope.carbon:[[@LINE+4]]:10: error: name `F` not found [NameNotFound]
  // CHECK:STDERR:   return F();
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return F();
}

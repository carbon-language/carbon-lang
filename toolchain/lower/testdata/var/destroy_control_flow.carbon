// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/destroy_control_flow.carbon

// --- destructible.carbon

library "[[@TEST_NAME]]";

class Destructible {
  // TODO: Implement Core.Destructor instead.
  impl as Core.Destroy {
    fn Op(self);
  }
};

fn Use(ref d: Destructible);

// --- if_then.carbon

library "[[@TEST_NAME]]";
import library "destructible";

// CHECK:STDERR: if_then.carbon:[[@LINE+4]]:17: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn IfThen(cond: bool) {
// CHECK:STDERR:                 ^~~~
// CHECK:STDERR:
fn IfThen(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  }
  Use(ref a);
}

// --- if_else.carbon

library "[[@TEST_NAME]]";
import library "destructible";

// CHECK:STDERR: if_else.carbon:[[@LINE+4]]:17: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn IfElse(cond: bool) {
// CHECK:STDERR:                 ^~~~
// CHECK:STDERR:
fn IfElse(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  } else {
    var c: Destructible = {};
    Use(ref c);
  }
  Use(ref a);
}

// --- if_return.carbon

library "[[@TEST_NAME]]";
import library "destructible";

// CHECK:STDERR: if_return.carbon:[[@LINE+4]]:19: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn IfReturn(cond: bool) {
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR:
fn IfReturn(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
    return;
  }
}

// --- todo_while_break.carbon

library "[[@TEST_NAME]]";
import library "destructible";

// CHECK:STDERR: todo_while_break.carbon:[[@LINE+4]]:21: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn WhileBreak(cond: bool) {
// CHECK:STDERR:                     ^~~~
// CHECK:STDERR:
fn WhileBreak(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      // TODO: We don't produce any destroy calls for this branch.
      break;
    }
  }
  Use(ref a);
}

// --- while_continue.carbon

library "[[@TEST_NAME]]";
import library "destructible";

// CHECK:STDERR: while_continue.carbon:[[@LINE+4]]:24: error: name `Core.Bool` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn WhileContinue(cond: bool) {
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
fn WhileContinue(cond: bool) {
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      // TODO: We don't produce any destroy calls for this branch.
      continue;
    }
  }
  Use(ref a);
}


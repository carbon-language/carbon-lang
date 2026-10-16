// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/access/friend.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/access/friend.carbon

// --- fail_non_friend.carbon
library "[[@TEST_NAME]]";

class C {
  private var i: i32;
}

fn F(c: C) -> i32 {
  // CHECK:STDERR: fail_non_friend.carbon:[[@LINE+3]]:10-12: error: cannot access private member `i` of type `C` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_non_friend.carbon:[[@LINE-5]]:15-20: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  return c.i;
}

// --- fail_todo_friend_function.carbon
library "[[@TEST_NAME]]";

class C;

fn F(c: C) -> i32;

class C {
  private var i: i32;
  // CHECK:STDERR: fail_todo_friend_function.carbon:[[@LINE+2]]:3-11: error: semantics TODO: `befriending a function not yet implemented` [SemanticsTodo]
  // CHECK:STDERR:
  friend F;
}

fn F(c: C) -> i32 {
  return c.i;
}

// --- friend_class.carbon
library "[[@TEST_NAME]]";

class Friend;

class C {
  private var i: i32;
  friend Friend;
}

class Friend {
  fn F(c: C) -> i32 {
    return c.i;
  }
}

// --- out_of_line_method.carbon
library "[[@TEST_NAME]]";

class Friend;

class C {
  private var i: i32;
  friend Friend;
}

class Friend {
  fn F(c: C) -> i32;
}

fn Friend.F(c: C) -> i32 {
  return c.i;
}

// --- friend_complete_class.carbon
library "[[@TEST_NAME]]";

class C;

class Friend {
  fn F(c: C) -> i32;
}

class C {
  private var i: i32;
  friend Friend;
}

fn Friend.F(c: C) -> i32 {
  return c.i;
}

// --- friend_complete_class_fn.carbon
library "[[@TEST_NAME]]";

class Friend {
  fn F() -> i32;
}

class C {
  private var i: i32;
  private fn Make() -> C { return {.i = 0}; }
  friend Friend;
}

fn Friend.F() -> i32 {
  return C.Make().i;
}

// --- friend_parameterized_class.carbon
library "[[@TEST_NAME]]";

class Friend(T: type);

class C {
  private var i: i32;
  friend Friend;
}

class Friend(T: type) {
  fn F(c: C) -> i32 {
    return c.i;
  }
}

// --- fail_friend_var.carbon
library "[[@TEST_NAME]]";

var v: i32;

class C {
  private var i: i32;
  // CHECK:STDERR: fail_friend_var.carbon:[[@LINE+2]]:10: error: friend declaration names entity that is not a name scope [FriendNotNameScope]
  // CHECK:STDERR:
  friend v;
}

// --- friend_interface.carbon
library "[[@TEST_NAME]]";

interface I;

class C {
  private var i: i32 = 0;
  friend I;
}

interface I {
  default fn F(c: C) -> i32 {
    return c.i;
  }
}

// --- fail_friend_not_in_class.carbon
library "[[@TEST_NAME]]";

class C;

interface I {
  // CHECK:STDERR: fail_friend_not_in_class.carbon:[[@LINE+3]]:3-11: error: `friend` declaration in non-class scope [FriendNotAllowedInScope]
  // CHECK:STDERR: fail_friend_not_in_class.carbon:[[@LINE-2]]:1-13: note: appears in this definition [FriendEnclosingScope]
  // CHECK:STDERR:
  friend C;
}

// --- fail_friend_not_constant.carbon
library "[[@TEST_NAME]]";

class C {
  var v: i32;
  // CHECK:STDERR: fail_friend_not_constant.carbon:[[@LINE+2]]:10: error: friend declaration names entity that is not a name scope [FriendNotNameScope]
  // CHECK:STDERR:
  friend v;
}

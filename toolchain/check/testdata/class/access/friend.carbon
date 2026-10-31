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
  // CHECK:STDERR:        ╭─┤ fail_non_friend.carbon:[[@LINE+11]]:10
  // CHECK:STDERR:        │   private var i: i32;
  // CHECK:STDERR:        ·               ───┬──
  // CHECK:STDERR:        ·                  ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return c.i;
  // CHECK:STDERR:        ·          ━┳━
  // CHECK:STDERR:        ·           ┗━━┫ error: cannot access private member `i` of type `C`
  // CHECK:STDERR:        ·              ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return c.i;
}

// --- fail_todo_friend_function.carbon
library "[[@TEST_NAME]]";

class C;

fn F(c: C) -> i32;

class C {
  private var i: i32;
  // CHECK:STDERR:        ╭─┤ fail_todo_friend_function.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   friend F;
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: semantics TODO: `befriending a function not yet implemented`
  // CHECK:STDERR:        ·          ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_friend_var.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   friend v;
  // CHECK:STDERR:        ·          ┳
  // CHECK:STDERR:        ·          ┗━━┫ error: friend declaration names entity that is not a name scope
  // CHECK:STDERR:        ·             ┃        [FriendNotNameScope]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_friend_not_in_class.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   friend C;
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: `friend` declaration in non-class scope
  // CHECK:STDERR:        ·          ┃        [FriendNotAllowedInScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend C;
}

// --- fail_friend_not_constant.carbon
library "[[@TEST_NAME]]";

class C {
  var v: i32;
  // CHECK:STDERR:        ╭─┤ fail_friend_not_constant.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   friend v;
  // CHECK:STDERR:        ·          ┳
  // CHECK:STDERR:        ·          ┗━━┫ error: friend declaration names entity that is not a name scope
  // CHECK:STDERR:        ·             ┃        [FriendNotNameScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend v;
}

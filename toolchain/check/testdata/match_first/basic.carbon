// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ^ARGS: -v compile %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/match_first/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/match_first/basic.carbon

// --- empty.carbon
library "[[@TEST_NAME]]";

match_first {}

// --- todo_fail_class_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  class C;
}

// --- todo_fail_interface_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  interface Z;
}

// --- todo_fail_constraint_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  constraint N;
}

// --- todo_fail_function_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  fn F();
}

// --- todo_fail_final_inside.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  // TODO: Can't write `final` inside the match first.
  final impl () as Z {}
}

// --- decl.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
}
impl () as Z {}

// --- definition.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z {}
}

// --- redecl.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z;
match_first {
  impl () as Z {}
}

// --- inside_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  match_first {
    impl Self as Z {}
  }
}

// --- inside_function.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  match_first {
    impl () as Z {}
  }
}

// --- fail_inside_interface.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  // CHECK:STDERR: fail_inside_interface.carbon:[[@LINE+7]]:3: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR:   match_first {
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_inside_interface.carbon:[[@LINE-4]]:1: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR: interface Y {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  match_first {
    impl Self as Z {}
  }
}

// --- fail_inside_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint N {
  // CHECK:STDERR: fail_inside_constraint.carbon:[[@LINE+7]]:3: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR:   match_first {
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_inside_constraint.carbon:[[@LINE-4]]:1: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR: constraint N {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  match_first {
    impl Self as Z {}
  }
}

// --- fail_inside_match_first.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  // CHECK:STDERR: fail_inside_match_first.carbon:[[@LINE+7]]:3: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR:   match_first {
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_inside_match_first.carbon:[[@LINE-4]]:1: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR: match_first {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  match_first {
    impl () as Z {}
  }
}

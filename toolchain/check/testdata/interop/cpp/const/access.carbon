// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/const/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/const/access.carbon

// --- const_member_access.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {
  int x;
};

const S& Get();
''';

fn F() {
  Cpp.Get().x;
}

// --- fail_const_member_assign.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {
  int x;
};

const S& Get();
''';

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_const_member_assign.carbon:[[@LINE+17]]:3
  // CHECK:STDERR: ->     │   Cpp.Get().x = 123;
  // CHECK:STDERR:        ·   ━━━━━┳━━━━━
  // CHECK:STDERR:        ·        ┗━━┫ error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_const_member_assign.carbon:[[@LINE+11]]:3
  // CHECK:STDERR: ->     │   Cpp.Get().x = 123;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert expression of type `Core.IntLiteral` to
  // CHECK:STDERR:        ·   │  ┃        `const i32`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·   ╰────┤ type `Core.IntLiteral` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(const i32)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.Get().x = 123;
}

// --- fail_const_member_assign_cast.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
struct S {
  int x;
};

const S& Get();
''';

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_const_member_assign_cast.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   Cpp.Get().x = 123 as i32;
  // CHECK:STDERR:        ·   ━━━━━┳━━━━━
  // CHECK:STDERR:        ·        ┗━━┫ error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.Get().x = 123 as i32;
}

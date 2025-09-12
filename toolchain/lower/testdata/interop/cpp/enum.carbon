// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/enum.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/enum.carbon

// TODO: Tests marked as `fail_todo_5891_` to fixed as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.

// --- enum_member.h

struct C {
  enum E : short {
    a,
    b = 7,
    c
  };

  static void F(E);
};

// --- fail_todo_5891_pass_as_arg.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn Pass() {
  Cpp.C.F(Cpp.C.a);
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.b);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.b);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.b);
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.c);
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.a);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.a);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.E.a);
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.b);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.b);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.E.b);
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_pass_as_arg.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.c);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.E.c);
}

// --- fail_todo_5891_convert.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn TakeI16(n: i16);

fn PassEnum() {
  TakeI16(Cpp.C.b as i16);
}

fn ConvertEnumToI16(e: Cpp.C.E) {
  TakeI16(e as i16);
}

fn ConvertI16ToEnum(n: i16) {
  // CHECK:STDERR: fail_todo_5891_convert.carbon:[[@LINE+7]]:3: error: no matching function for call to `F` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.C.F(n as Cpp.C.E);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_convert.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.C.F(n as Cpp.C.E);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(n as Cpp.C.E);
}

// --- bitmask.h

enum Bits {
  A = 1 << 0,
  B = 1 << 1,
  C = 1 << 2,
};

void Take(Bits b);

// --- fail_todo_use_5891_bitmask.carbon

library "[[@TEST_NAME]]";

import Cpp library "bitmask.h";

// TODO: Once we support interop with C++-defined operators, we should be able
// to use `|` below rather than declaring our own builtin.
fn BitOr(a: Cpp.Bits, b: Cpp.Bits) -> Cpp.Bits = "int.or";

fn Call() {
  // CHECK:STDERR: fail_todo_use_5891_bitmask.carbon:[[@LINE+7]]:3: error: no matching function for call to `Take` [CppOverloadingNoViableFunctionFound]
  // CHECK:STDERR:   Cpp.Take(BitOr(Cpp.A, Cpp.C));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_use_5891_bitmask.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.Take(BitOr(Cpp.A, Cpp.C));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Take(BitOr(Cpp.A, Cpp.C));
}

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

// --- enum_member.h

struct C {
  enum E : short {
    a,
    b = 7,
    c
  };

  static void F(E);
};

// --- fail_todo_pass_as_arg.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn Pass() {
  Cpp.C.F(Cpp.C.a);
  Cpp.C.F(Cpp.C.a);
  Cpp.C.F(Cpp.C.b);
  Cpp.C.F(Cpp.C.c);
  Cpp.C.F(Cpp.C.E.a);
  Cpp.C.F(Cpp.C.E.b);
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE+28]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.c);
  // CHECK:STDERR:           ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-5]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.b);
  // CHECK:STDERR:           ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-10]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.E.a);
  // CHECK:STDERR:           ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-15]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.c);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-20]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.b);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-25]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.a);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_pass_as_arg.carbon:[[@LINE-30]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(Cpp.C.a);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR:
  Cpp.C.F(Cpp.C.E.c);
}

// --- fail_todo_convert.carbon

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
  // CHECK:STDERR: fail_todo_convert.carbon:[[@LINE+4]]:11: error: cannot access member of interface `Core.Destroy` in type `Cpp.E` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   Cpp.C.F(n as Cpp.C.E);
  // CHECK:STDERR:           ^~~~~~~~~~~~
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

// --- use_bitmask.carbon

library "[[@TEST_NAME]]";

import Cpp library "bitmask.h";

// TODO: Once we support interop with C++-defined operators, we should be able
// to use `|` below rather than declaring our own builtin.
fn BitOr(a: Cpp.Bits, b: Cpp.Bits) -> Cpp.Bits = "int.or";

fn Call() {
  Cpp.Take(BitOr(Cpp.A, Cpp.C));
}

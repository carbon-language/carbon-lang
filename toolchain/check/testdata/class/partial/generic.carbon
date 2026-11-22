// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial/generic.carbon

// --- base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](unused p: partial T*) {}
//@dump-sem-ir-end

fn G(b: partial B*) {
  //@dump-sem-ir-begin
  F(b);
  //@dump-sem-ir-end
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class A {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](unused p: partial T*) {}
//@dump-sem-ir-end

fn G(a: partial A*) {
  //@dump-sem-ir-begin
  F(a);
  //@dump-sem-ir-end
}

// --- convert.carbon
library "[[@TEST_NAME]]";

base class B {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}
//@dump-sem-ir-end

fn Test(ref b: partial B) {
  //@dump-sem-ir-begin
  F(ref b);
  //@dump-sem-ir-end
}

// --- fail_period_self_unconstrained.carbon
library "[[@TEST_NAME]]";

interface I {}

fn F[unused T: type where partial .Self impls I]() {}

// --- fail_unconstrained_member_access.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op(ref self);
}

fn F[T: Core.Extendable](ref t: T) {
  (t as partial T).(I.Op)();
}

// --- fail_final_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn F[T: Core.Extendable](unused t: T) {}

fn G(c: C) {
  F(c);
}


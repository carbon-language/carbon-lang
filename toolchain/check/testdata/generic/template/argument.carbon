// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/argument.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/argument.carbon

// --- constant_type_argument.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused x: T) {}

class X {}

fn G(x: X) {
  F((), ());
  F({}, {});
  F(X, {});
  F(type, X);
  F(X, x);
}

// --- forward_template_type_to_template.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused x: T) {}

fn ForwardTemplate(template T: type, x: T) {
  F(T, x);
}

fn CallForwardTemplate() {
  ForwardTemplate({}, {});
}

// --- forward_generic_type_to_template.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused x: T) {}

fn ForwardGeneric(generic T: type, x: T) {
  F(T, x);
}

fn CallForwardGeneric() {
  ForwardGeneric({}, {});
}

// --- constant_nontype_argument.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused template X: T) {}

class X {}

fn G() {
  F((), ());
  F({}, {});
  F(X, {});
  F(type, X);
}

// --- addr_of_dependent_type.carbon
library "[[@TEST_NAME]]";

class C {};
impl C as Core.Copy {
  fn Op(unused self) -> C { return {} as C; }
}
class D(template T: type) {
  fn F(ref self) -> C* { return &self.c; }
  var c: C;
}

var d: D(C) = {.c = {} as C};
let e: C* = d.F();

// --- fail_addr_of_non_addressable_dependent_type.carbon
library "[[@TEST_NAME]]";

class C {};
impl C as Core.Copy {
  fn Op(unused self) -> C { return {} as C; }
}
class D(template T: type) {
  // Note thae missing `ref` qualifier on the `self` parameter, which makes
  // the expression `self.c` non-addressable.
  // CHECK:STDERR:        ╭─┤ fail_addr_of_non_addressable_dependent_type.carbon:[[@LINE+12]]:29
  // CHECK:STDERR: ->     │   fn F(self) -> C* { return &self.c; }
  // CHECK:STDERR:        ·                             ┳━━━━━━
  // CHECK:STDERR:        ·                             ┗━━┫ error: cannot take the address of non-reference
  // CHECK:STDERR:        ·                                ┃        expression
  // CHECK:STDERR:        ·                                ┃        [AddrOfNonRef]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ let e: C* = d.F();
  // CHECK:STDERR:        ·             ─┬─
  // CHECK:STDERR:        ·              ╰──┤ in monomorphization of `F(C)` here [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  fn F(self) -> C* { return &self.c; }
  var c: C;
}

var d: D(C) = {.c = {} as C};
let e: C* = d.F();

// --- fail_non_constant_type_argument.carbon
library "[[@TEST_NAME]]";

class Y {}
let T: type = Y;

fn F(template T: type, unused y: T) {}

fn H(y: Y) {
  // CHECK:STDERR:        ╭─┤ fail_non_constant_type_argument.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F(template T: type, unused y: T) {}
  // CHECK:STDERR:        ·      ────────┬───────
  // CHECK:STDERR:        ·              ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                 │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(T, y);
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: argument for generic parameter is not a compile-time constant
  // CHECK:STDERR:        ·         ┃        [CompTimeArgumentNotConstant]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(T, y);
}

// --- fail_non_constant_nontype_argument.carbon
library "[[@TEST_NAME]]";

class Y {}

fn F(unused template X: Y) {}

fn H(y: Y) {
  // CHECK:STDERR:        ╭─┤ fail_non_constant_nontype_argument.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F(unused template X: Y) {}
  // CHECK:STDERR:        ·             ──────┬──────
  // CHECK:STDERR:        ·                   ╰──┤ initializing generic parameter `X` declared here
  // CHECK:STDERR:        ·                      │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(y);
  // CHECK:STDERR:        ·   ━━┳━
  // CHECK:STDERR:        ·     ┗━━┫ error: argument for generic parameter is not a compile-time constant
  // CHECK:STDERR:        ·        ┃        [CompTimeArgumentNotConstant]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(y);
}

// --- template_signature_type_matching.carbon
library "[[@TEST_NAME]]";

class C(T: type) {};
fn F(template T: type, c: C(T)) -> C(T);
fn F(template T: type, c: C(T)) -> C(T) { return c; }

// --- fail_template_signature_type_mismatch.carbon
library "[[@TEST_NAME]]";

class C(T: type) {};
class D(T: type) {};
fn F(template T: type, c: C(T)) -> C(T);
// CHECK:STDERR:        ╭─┤ fail_template_signature_type_mismatch.carbon:[[@LINE+12]]:34
// CHECK:STDERR:        │ fn F(template T: type, c: C(T)) -> C(T);
// CHECK:STDERR:        ·                           ┬
// CHECK:STDERR:        ·                           ╰──┤ comparing with previous declaration here
// CHECK:STDERR:        ·                              │ [RedeclParamSyntaxPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F(template T: type, unused c: D(T)) -> C(T) { return {} as D(T); }
// CHECK:STDERR:        ·                                  ┳
// CHECK:STDERR:        ·                                  ┗━━┫ error: redeclaration syntax differs here
// CHECK:STDERR:        ·                                     ┃        [RedeclParamSyntaxDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(template T: type, unused c: D(T)) -> C(T) { return {} as D(T); }

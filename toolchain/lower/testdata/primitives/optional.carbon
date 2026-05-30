// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is an integration test for the Core.Optional in the prelude, so use the
// real prelude.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/optional.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/optional.carbon
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:46:14: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Convert[self: Self]() -> T {
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:54:12: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Equal[self: Self](other: Char) -> bool = "int.eq";
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:55:15: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn NotEqual[self: Self](other: Char) -> bool = "int.neq";
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:59:11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Less[self: Self](other: Char) -> bool = "int.less";
// CHECK:STDERR:           ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:60:23: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn LessOrEquivalent[self: Self](other: Char) -> bool = "int.less_eq";
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:61:14: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Greater[self: Self](other: Char) -> bool = "int.greater";
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:62:26: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn GreaterOrEquivalent[self: Self](other: Char) -> bool = "int.greater_eq";
// CHECK:STDERR:                          ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:66:12: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Equal[self: Self](other: Char) -> bool = "int.eq";
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:67:15: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn NotEqual[self: Self](other: Char) -> bool = "int.neq";
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:71:11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Less[self: Self](other: Char) -> bool = "int.less";
// CHECK:STDERR:           ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:72:23: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn LessOrEquivalent[self: Self](other: Char) -> bool = "int.less_eq";
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:73:14: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Greater[self: Self](other: Char) -> bool = "int.greater";
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:74:26: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn GreaterOrEquivalent[self: Self](other: Char) -> bool = "int.greater_eq";
// CHECK:STDERR:                          ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:78:12: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Equal[self: Char](other: Char) -> bool = "int.eq";
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:79:15: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn NotEqual[self: Char](other: Char) -> bool = "int.neq";
// CHECK:STDERR:               ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:83:11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Less[self: Char](other: Char) -> bool = "int.less";
// CHECK:STDERR:           ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:84:23: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn LessOrEquivalent[self: Char](other: Char) -> bool = "int.less_eq";
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:85:14: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Greater[self: Char](other: Char) -> bool = "int.greater";
// CHECK:STDERR:              ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:86:26: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn GreaterOrEquivalent[self: Char](other: Char) -> bool = "int.greater_eq";
// CHECK:STDERR:                          ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:98:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Self](other: T) -> Self {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:106:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Self](other: Char) -> Char {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:114:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Self](other: T) -> Self {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:121:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Self](other: Char) -> i32 {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:127:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Char](other: Char) -> i32 {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/char.carbon:133:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Char](other: Char) -> i32 {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:50:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR:   fn Op[self: Self]() -> Self {
// CHECK:STDERR:         ^~~~~~~~~~
// CHECK:STDERR:

// --- convert.carbon
library "[[@TEST_NAME]]";

fn Convert(o: Core.Optional(i32*)) -> Core.Optional(i32) {
  if (o.HasValue()) {
    return *o.Get();
  }
  return Core.Optional(i32).None();
}

fn AddOrRemoveConst(a: i32, b: const i32) {
  var _: Core.Optional(i32) = a;
  var _: const Core.Optional(i32) = a;
  var _: Core.Optional(const i32) = a;
  var _: const Core.Optional(const i32) = a;
  var _: Core.Optional(i32) = b;
  var _: const Core.Optional(i32) = b;
  var _: Core.Optional(const i32) = b;
  var _: const Core.Optional(const i32) = b;
}

// --- copy.carbon
library "[[@TEST_NAME]]";

class Copyable {
  impl as Core.Copy {
    // CHECK:STDERR: copy.carbon:[[@LINE+4]]:11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
    // CHECK:STDERR:     fn Op[self: Self]() -> Self;
    // CHECK:STDERR:           ^~~~~~~~~~
    // CHECK:STDERR:
    fn Op[self: Self]() -> Self;
  }
}

fn F(c: Core.Optional(Copyable)) -> Core.Optional(Copyable) {
  // Should only perform one copy.
  return c;
}

// --- copy_pointer.carbon
library "[[@TEST_NAME]]";

class C {}

fn F(c: Core.Optional(C*)) -> Core.Optional(C*) {
  return c;
}


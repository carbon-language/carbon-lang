// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is testing a facet type that combines two interfaces across multiple
// import boundaries.
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/import_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/import_facet.carbon
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

// --- a.carbon
library "[[@TEST_NAME]]";

class C(T:! Core.Copy & Core.UnformedInit & Core.Destroy) {
  fn GetC(unused self) -> T {
    var value: T;
    return value;
  }
}

// --- b.carbon
library "[[@TEST_NAME]]";
export import library "a";

interface I {
  let T:! Core.Copy & Core.UnformedInit & Core.Destroy;
  fn GetI(self) -> C(T);
  // CHECK:STDERR: b.carbon:[[@LINE+15]]:18: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:   fn GetI[unused self: Self]() -> C(T);
  // CHECK:STDERR:                  ^~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: b.carbon:[[@LINE+11]]:3: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
  // CHECK:STDERR:   fn GetI[unused self: Self]() -> C(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: b.carbon:[[@LINE-8]]:3: note: previously declared without implicit parameter list [RedeclParamListPrevious]
  // CHECK:STDERR:   fn GetI(self) -> C(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: b.carbon:[[@LINE+4]]:18: error: `unused` modifier without a definition [UnusedModifierWithoutDefinition]
  // CHECK:STDERR:   fn GetI[unused self: Self]() -> C(T);
  // CHECK:STDERR:                  ^~~~~~~~~~
  // CHECK:STDERR:
  fn GetI[unused self: Self]() -> C(T);
  // CHECK:STDERR: b.carbon:[[@LINE+11]]:11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:   fn GetI[self: Self]() -> C(T);
  // CHECK:STDERR:           ^~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: b.carbon:[[@LINE+7]]:3: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
  // CHECK:STDERR:   fn GetI[self: Self]() -> C(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: b.carbon:[[@LINE-24]]:3: note: previously declared without implicit parameter list [RedeclParamListPrevious]
  // CHECK:STDERR:   fn GetI(self) -> C(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn GetI[self: Self]() -> C(T);
}

// --- c.carbon
library "[[@TEST_NAME]]";
export import library "b";

class D(U:! type) {
  impl as I where .T = U* {
    fn GetI(unused self) -> C(U*) {
      return {};
    }
  }
}

// --- d.carbon
library "[[@TEST_NAME]]";
import library "c";

fn F(d: D(i32)) -> i32* {
  return d.(I.GetI)().GetC();
}


// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/pointer.carbon
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

// ============================================================================
// Non-null pointers
// ============================================================================

// --- nonnull.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto TakePtr(C* _Nonnull) -> void;
auto ReturnPtr() -> C* _Nonnull;

auto TakePtrWithThunk(C* _Nonnull, int = 0) -> void;
auto ReturnPtrWithThunk(int = 0) -> C* _Nonnull;
''';

fn PassPtr(p: Cpp.C*) {
  Cpp.TakePtr(p);
}

fn ReturnPtr() -> Cpp.C* {
  return Cpp.ReturnPtr();
}

fn PassPtrWithThunk(p: Cpp.C*) {
  Cpp.TakePtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Cpp.C* {
  return Cpp.ReturnPtrWithThunk();
}

// ============================================================================
// Nullable pointers
// ============================================================================

// --- nullable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto TakePtr(C*) -> void;
auto ReturnPtr() -> C*;

auto TakePtrWithThunk(C*, int = 0) -> void;
auto ReturnPtrWithThunk(int = 0) -> C*;
''';

fn PassPtr(_: Core.Optional(Cpp.C*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakePtr(p);
}

fn PassNonnullPtr(p: Cpp.C*) {
  // TODO: The `value_of_initializer` optimization doesn't get used here, so we
  // unnecessarily round-trip this value through memory.
  Cpp.TakePtr(p);
}

fn ReturnPtr() -> Core.Optional(Cpp.C*) {
  return Cpp.ReturnPtr();
}

fn PassPtrWithThunk(_: Core.Optional(Cpp.C*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakePtrWithThunk(p);
}

fn PassNonnullPtrWithThunk(p: Cpp.C*) {
  Cpp.TakePtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Core.Optional(Cpp.C*) {
  return Cpp.ReturnPtrWithThunk();
}

// --- add_const.carbon

import Cpp inline '''
class C {};
C *make();
void take(const C*);
''';

fn Convert() {
  Cpp.take(Cpp.make());
}


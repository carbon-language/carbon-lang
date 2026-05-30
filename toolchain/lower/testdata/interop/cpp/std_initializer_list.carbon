// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list.carbon
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

// --- initializer_list_pointer_pointer.h

namespace std {
  using size_t = decltype(sizeof(0));

  template <typename T> class initializer_list {
   public:
    initializer_list() {}

   private:
    T* begin_ = nullptr;
    T* end_ = nullptr;
  };
}

struct vector_like {
  vector_like(std::initializer_list<int> list) {}
  ~vector_like() {}
};

// --- initializer_list_pointer_size.h

namespace std {
  using size_t = decltype(sizeof(0));

  template <typename T> class initializer_list {
   public:
    initializer_list() {}

   private:
    T* begin_ = nullptr;
    size_t size_ = 0;
  };
}

struct vector_like {
  vector_like(std::initializer_list<int> list) {}
  ~vector_like() {}
};

// --- nontrivial_dtor.h

struct nontrivial_dtor {
  nontrivial_dtor() {}
  ~nontrivial_dtor() {}
};

// --- use_pointer_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "initializer_list_pointer_pointer.h";
import Cpp library "nontrivial_dtor.h";

// TODO: When we support more precise modeling of temporary lifetimes, ensure
// that this is called *before* the elements of the underlying array are
// destroyed in the non-vector-like cases.
fn WithinLifetime();

fn InitDirectly() {
  var _: Cpp.std.initializer_list(i32) = (1, 2, 3);
  WithinLifetime();
}

fn InitVectorLike() {
  var _: Cpp.vector_like = (1, 2, 3);
  WithinLifetime();
}

fn InitNontrivialDtor() {
  var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  WithinLifetime();
}

// --- use_pointer_size.carbon

library "[[@TEST_NAME]]";

import Cpp library "initializer_list_pointer_size.h";
import Cpp library "nontrivial_dtor.h";

// TODO: When we support more precise modeling of temporary lifetimes, ensure
// that this is called *before* the elements of the underlying array are
// destroyed in the non-vector-like cases.
fn WithinLifetime();

fn InitDirectly() {
  var _: Cpp.std.initializer_list(i32) = (1, 2, 3);
  WithinLifetime();
}

fn InitVectorLike() {
  var _: Cpp.vector_like = (1, 2, 3);
  WithinLifetime();
}

fn InitNontrivialDtor() {
  var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  WithinLifetime();
}


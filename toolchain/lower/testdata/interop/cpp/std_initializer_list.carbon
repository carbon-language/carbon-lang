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
  // CHECK:STDERR: use_pointer_pointer.carbon:[[@LINE+11]]:58: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  // CHECK:STDERR:                                                          ^~~~~~~~~~~~
  // CHECK:STDERR: use_pointer_pointer.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./nontrivial_dtor.h:3:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   nontrivial_dtor() {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR: use_pointer_pointer.carbon:[[@LINE+4]]:3: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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
  // CHECK:STDERR: use_pointer_size.carbon:[[@LINE+11]]:58: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  // CHECK:STDERR:                                                          ^~~~~~~~~~~~
  // CHECK:STDERR: use_pointer_size.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./nontrivial_dtor.h:3:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   nontrivial_dtor() {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR: use_pointer_size.carbon:[[@LINE+4]]:3: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  WithinLifetime();
}


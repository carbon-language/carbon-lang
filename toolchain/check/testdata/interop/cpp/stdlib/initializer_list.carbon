// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/stdlib/initializer_list.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/stdlib/initializer_list.carbon

// --- initializer_list.h

namespace std {
  using size_t = __SIZE_TYPE__;

  template<typename T> class initializer_list {
   public:
    initializer_list() {}
    const T* begin() { return data_; }
    const T* end() { return data_ + size_; }
    size_t size() { return size_; }

   private:
    const T* data_ = nullptr;
    size_t size_ = 0;
  };
}

auto Consume(std::initializer_list<int> il) -> void;

struct InitListConstructor {
  InitListConstructor(std::initializer_list<int> il) {}
};

// --- convert.carbon

library "[[@TEST_NAME]]";

import Cpp library "initializer_list.h";

//@dump-sem-ir-begin
fn F() {
  // CHECK:STDERR: convert.carbon:[[@LINE+4]]:42: error: semantics TODO: `std::initializer_list construction` [SemanticsTodo]
  // CHECK:STDERR:   let _: Cpp.std.initializer_list(i32) = (1, 2, 3);
  // CHECK:STDERR:                                          ^~~~~~~~~
  // CHECK:STDERR:
  let _: Cpp.std.initializer_list(i32) = (1, 2, 3);

  // CHECK:STDERR: convert.carbon:[[@LINE+8]]:15: error: semantics TODO: `std::initializer_list construction` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.Consume((1, 2, 3));
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR: convert.carbon:[[@LINE-13]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./initializer_list.h:18:41: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto Consume(std::initializer_list<int> il) -> void;
  // CHECK:STDERR:                                         ^
  // CHECK:STDERR:
  Cpp.Consume((1, 2, 3));

  // CHECK:STDERR: convert.carbon:[[@LINE+8]]:36: error: semantics TODO: `std::initializer_list construction` [SemanticsTodo]
  // CHECK:STDERR:   let _: Cpp.InitListConstructor = (1, 2, 3);
  // CHECK:STDERR:                                    ^~~~~~~~~
  // CHECK:STDERR: convert.carbon:[[@LINE-23]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./initializer_list.h:21:50: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   InitListConstructor(std::initializer_list<int> il) {}
  // CHECK:STDERR:                                                  ^
  // CHECK:STDERR:
  let _: Cpp.InitListConstructor = (1, 2, 3);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.type: type = cpp_type_template_type initializer_list [concrete]
// CHECK:STDOUT:   %initializer_list.template: %initializer_list.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list: type = class_type @initializer_list [concrete]
// CHECK:STDOUT:   %pattern_type.22f: type = pattern_type %initializer_list [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2, %int_3) [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.type: type = cpp_overload_set_type @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.82f: type = ptr_type %initializer_list [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk.type: type = fn_type @Consume__carbon_thunk [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk: %Consume__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor: type = class_type @InitListConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.92d: type = pattern_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %ptr.79d: type = ptr_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.type: type = fn_type @InitListConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk: %InitListConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.type: type = fn_type @InitListConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor: %InitListConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .std = %std
// CHECK:STDOUT:     .Consume = %Consume.cpp_overload_set.value
// CHECK:STDOUT:     .InitListConstructor = %InitListConstructor.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %std: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .initializer_list = %initializer_list.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.template: %initializer_list.type = struct_value () [concrete = constants.%initializer_list.template]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %Consume__carbon_thunk.decl: %Consume__carbon_thunk.type = fn_decl @Consume__carbon_thunk [concrete = constants.%Consume__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitListConstructor.decl: type = class_decl @InitListConstructor [concrete = constants.%InitListConstructor] {} {}
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.decl: %InitListConstructor__carbon_thunk.type = fn_decl @InitListConstructor__carbon_thunk [concrete = constants.%InitListConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.22f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc12: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc12_50.1: %tuple.type = tuple_literal (%int_1.loc12, %int_2.loc12, %int_3.loc12) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc12_50.2: %initializer_list = converted %.loc12_50.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %_.loc12: %initializer_list = value_binding _, <error> [concrete = <error>]
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %Consume.cpp_overload_set.type = name_ref Consume, imports.%Consume.cpp_overload_set.value [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc22: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc22_23.1: %tuple.type = tuple_literal (%int_1.loc22, %int_2.loc22, %int_3.loc22) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc22_23.2: %initializer_list = converted %.loc22_23.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc22: %ptr.82f = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %Consume__carbon_thunk.call: init %empty_tuple.type = call imports.%Consume__carbon_thunk.decl(%addr.loc22)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc32: %pattern_type.92d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc32: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc32_44.1: %tuple.type = tuple_literal (%int_1.loc32, %int_2.loc32, %int_3.loc32) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc32_13: type = splice_block %InitListConstructor.ref [concrete = constants.%InitListConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InitListConstructor.ref: type = name_ref InitListConstructor, imports.%InitListConstructor.decl [concrete = constants.%InitListConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc32_44.2: ref %InitListConstructor = temporary_storage
// CHECK:STDOUT:   %.loc32_44.3: %initializer_list = converted %.loc32_44.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc32_44.1: %ptr.82f = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc32_44.2: %ptr.79d = addr_of %.loc32_44.2
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InitListConstructor__carbon_thunk.decl(%addr.loc32_44.1, %addr.loc32_44.2)
// CHECK:STDOUT:   %.loc32_44.4: init %InitListConstructor to %.loc32_44.2 = in_place_init %InitListConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc32_44.5: init %InitListConstructor = converted %.loc32_44.1, %.loc32_44.4
// CHECK:STDOUT:   %.loc32_44.6: ref %InitListConstructor = temporary %.loc32_44.2, %.loc32_44.5
// CHECK:STDOUT:   %.loc32_44.7: %InitListConstructor = acquire_value %.loc32_44.6
// CHECK:STDOUT:   %_.loc32: %InitListConstructor = value_binding _, %.loc32_44.7
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc32_44.6, constants.%InitListConstructor.cpp_destructor
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.call: init %empty_tuple.type = call %InitListConstructor.cpp_destructor.bound(%.loc32_44.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc32(%self.param: %InitListConstructor) = "no_op";
// CHECK:STDOUT:

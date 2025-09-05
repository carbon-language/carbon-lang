// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/stdlib/string_view.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/stdlib/string_view.carbon

// --- string_view.h

namespace std {
  using size_t = __SIZE_TYPE__;

  inline namespace __1 {
    template<typename T> struct char_traits {};

    template<typename CharT, typename Traits = char_traits<CharT>>
    class basic_string_view {
     public:
      basic_string_view() = default;
      size_t size() const { return size_; }

     private:
      const CharT* data_;
      size_t size_;
    };

    using string_view = basic_string_view<char>;
  }
}

auto Consume(std::string_view sv) -> void;
auto Produce() -> std::string_view;

// --- fail_todo_import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "string_view.h";

//@dump-sem-ir-begin
fn F() {
  // CHECK:STDERR: fail_todo_import_multiple.carbon:[[@LINE+7]]:3: error: call arg of type `Core.String` is not supported [CallArgTypeNotSupported]
  // CHECK:STDERR:   Cpp.Consume("hello");
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_multiple.carbon:[[@LINE+4]]:3: note: in call to a Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.Consume("hello");
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Consume("hello");
}
//@dump-sem-ir-end

//@dump-sem-ir-begin
fn G() -> str {
  return Cpp.Produce();
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %String: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %.f1f: type = overloaded_cpp_fn_type @Produce [concrete]
// CHECK:STDOUT:   %empty_struct.aeb: %.f1f = struct_value () [concrete]
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "hello" [concrete]
// CHECK:STDOUT:   %int_5: %u64 = int_value 5 [concrete]
// CHECK:STDOUT:   %String.val: %String = struct_value (%str, %int_5) [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %String [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b53: type = overloaded_cpp_fn_type @Produce__carbon_thunk [concrete]
// CHECK:STDOUT:   %empty_struct.848: %.b53 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.85f: type = ptr_type %String [concrete]
// CHECK:STDOUT:   %Produce__carbon_thunk.type: type = fn_type @Produce__carbon_thunk [concrete]
// CHECK:STDOUT:   %Produce__carbon_thunk: %Produce__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %String.as.Destroy.impl.Op.type: type = fn_type @String.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %String.as.Destroy.impl.Op: %String.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Consume = %.345
// CHECK:STDOUT:     .Produce = %.9e9
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.345: %.f1f = overloaded_cpp_fn_decl @Produce [concrete = constants.%empty_struct.aeb]
// CHECK:STDOUT:   %.9e9: %.b53 = overloaded_cpp_fn_decl @Produce__carbon_thunk [concrete = constants.%empty_struct.848]
// CHECK:STDOUT:   %Produce__carbon_thunk.decl: %Produce__carbon_thunk.type = fn_decl @Produce__carbon_thunk [concrete = constants.%Produce__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.461 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.461 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref %String = out_param call_param0
// CHECK:STDOUT:     %return: ref %String = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %.f1f = name_ref Consume, imports.%.345 [concrete = constants.%empty_struct.aeb]
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "hello" [concrete = constants.%str]
// CHECK:STDOUT:   %int_5: %u64 = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %String.val: %String = struct_value (%str, %int_5) [concrete = constants.%String.val]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param: %String {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Produce.ref: %.b53 = name_ref Produce, imports.%.9e9 [concrete = constants.%empty_struct.848]
// CHECK:STDOUT:   %.loc20: ref %String = splice_block %return {}
// CHECK:STDOUT:   %addr.loc21: %ptr.85f = addr_of %.loc20
// CHECK:STDOUT:   %Produce__carbon_thunk.call: init %empty_tuple.type = call imports.%Produce__carbon_thunk.decl(%addr.loc21)
// CHECK:STDOUT:   %.loc21: init %String = in_place_init %Produce__carbon_thunk.call, %.loc20
// CHECK:STDOUT:   %String.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc20, constants.%String.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc20: %ptr.85f = addr_of %.loc20
// CHECK:STDOUT:   %String.as.Destroy.impl.Op.call: init %empty_tuple.type = call %String.as.Destroy.impl.Op.bound(%addr.loc20)
// CHECK:STDOUT:   return %.loc21 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

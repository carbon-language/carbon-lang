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

      constexpr bool starts_with(basic_string_view sv) const {
        return false;
      }

     private:
      const CharT* data_;
      size_t size_;
    };

    using string_view = basic_string_view<char>;
  }
}

auto Consume(std::string_view sv) -> void;
auto Produce() -> std::string_view;

struct DerivedFromStringView : std::string_view {};

// --- pass_and_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "string_view.h";

//@dump-sem-ir-begin
fn F() {
  Cpp.Consume("hello");
}
//@dump-sem-ir-end

//@dump-sem-ir-begin
fn G() -> str {
  return Cpp.Produce();
}
//@dump-sem-ir-end

// --- call_base_method.carbon

library "[[@TEST_NAME]]";

import Cpp library "string_view.h";

//@dump-sem-ir-begin
fn StarstWith(a: str, b: str) -> bool {
  return a.(Cpp.DerivedFromStringView.starts_with)(b);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- pass_and_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.type: type = cpp_overload_set_type @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %str.3b1: %ptr.3e8 = string_literal "hello" [concrete]
// CHECK:STDOUT:   %int_5: %u64 = int_value 5 [concrete]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str.3b1, %int_5) [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str.ee0 [concrete]
// CHECK:STDOUT:   %ptr.85f: type = ptr_type %str.ee0 [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk.type: type = fn_type @Consume__carbon_thunk [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk: %Consume__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.2ff: Core.Form = init_form %str.ee0, call_param0 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Produce.cpp_overload_set.type: type = cpp_overload_set_type @Produce.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Produce.cpp_overload_set.value: %Produce.cpp_overload_set.type = cpp_overload_set_value @Produce.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Produce__carbon_thunk.type: type = fn_type @Produce__carbon_thunk [concrete]
// CHECK:STDOUT:   %Produce__carbon_thunk: %Produce__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Consume = %Consume.cpp_overload_set.value
// CHECK:STDOUT:     .Produce = %Produce.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %Consume__carbon_thunk.decl: %Consume__carbon_thunk.type = fn_decl @Consume__carbon_thunk [concrete = constants.%Consume__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Produce.cpp_overload_set.value: %Produce.cpp_overload_set.type = cpp_overload_set_value @Produce.cpp_overload_set [concrete = constants.%Produce.cpp_overload_set.value]
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
// CHECK:STDOUT:     %str: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:     %.loc13_11.2: Core.Form = init_form %str, call_param0 [concrete = constants.%.2ff]
// CHECK:STDOUT:     %return.param: ref %str.ee0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %str.ee0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %Consume.cpp_overload_set.type = name_ref Consume, imports.%Consume.cpp_overload_set.value [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %str: %ptr.3e8 = string_literal "hello" [concrete = constants.%str.3b1]
// CHECK:STDOUT:   %int_5: %u64 = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %String.val: %str.ee0 = struct_value (%str, %int_5) [concrete = constants.%String.val]
// CHECK:STDOUT:   %.loc8: ref %str.ee0 = value_as_ref %String.val
// CHECK:STDOUT:   %addr: %ptr.85f = addr_of %.loc8
// CHECK:STDOUT:   %Consume__carbon_thunk.call: init %empty_tuple.type = call imports.%Consume__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %str.ee0 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Produce.ref: %Produce.cpp_overload_set.type = name_ref Produce, imports.%Produce.cpp_overload_set.value [concrete = constants.%Produce.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc13_11.1: ref %str.ee0 = splice_block %return.param {}
// CHECK:STDOUT:   %addr: %ptr.85f = addr_of %.loc13_11.1
// CHECK:STDOUT:   %Produce__carbon_thunk.call: init %empty_tuple.type = call imports.%Produce__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc14: init %str.ee0 to %.loc13_11.1 = mark_in_place_init %Produce__carbon_thunk.call
// CHECK:STDOUT:   return %.loc14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %str: type = class_type @String [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str [concrete]
// CHECK:STDOUT:   %.d29: Core.Form = init_form bool, call_param2 [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %StarstWith.type: type = fn_type @StarstWith [concrete]
// CHECK:STDOUT:   %StarstWith: %StarstWith.type = struct_value () [concrete]
// CHECK:STDOUT:   %DerivedFromStringView: type = class_type @DerivedFromStringView [concrete]
// CHECK:STDOUT:   %DerivedFromStringView.starts_with.cpp_overload_set.type: type = cpp_overload_set_type @DerivedFromStringView.starts_with.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DerivedFromStringView.starts_with.cpp_overload_set.value: %DerivedFromStringView.starts_with.cpp_overload_set.type = cpp_overload_set_value @DerivedFromStringView.starts_with.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.85f: type = ptr_type %str [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %starts_with__carbon_thunk.type: type = fn_type @starts_with__carbon_thunk [concrete]
// CHECK:STDOUT:   %starts_with__carbon_thunk: %starts_with__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DerivedFromStringView = %DerivedFromStringView.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DerivedFromStringView.decl: type = class_decl @DerivedFromStringView [concrete = constants.%DerivedFromStringView] {} {}
// CHECK:STDOUT:   %DerivedFromStringView.starts_with.cpp_overload_set.value: %DerivedFromStringView.starts_with.cpp_overload_set.type = cpp_overload_set_value @DerivedFromStringView.starts_with.cpp_overload_set [concrete = constants.%DerivedFromStringView.starts_with.cpp_overload_set.value]
// CHECK:STDOUT:   %starts_with__carbon_thunk.decl: %starts_with__carbon_thunk.type = fn_decl @starts_with__carbon_thunk [concrete = constants.%starts_with__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %StarstWith.decl: %StarstWith.type = fn_decl @StarstWith [concrete = constants.%StarstWith] {
// CHECK:STDOUT:     %a.patt: %pattern_type.461 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.461 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.461 = value_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.461 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_34.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc7_34.2: Core.Form = init_form %.loc7_34.1, call_param2 [concrete = constants.%.d29]
// CHECK:STDOUT:     %a.param: %str = value_param call_param0
// CHECK:STDOUT:     %str.loc7_18: type = type_literal constants.%str [concrete = constants.%str]
// CHECK:STDOUT:     %a: %str = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %str = value_param call_param1
// CHECK:STDOUT:     %str.loc7_26: type = type_literal constants.%str [concrete = constants.%str]
// CHECK:STDOUT:     %b: %str = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @StarstWith(%a.param: %str, %b.param: %str) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %str = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DerivedFromStringView.ref: type = name_ref DerivedFromStringView, imports.%DerivedFromStringView.decl [concrete = constants.%DerivedFromStringView]
// CHECK:STDOUT:   %starts_with.ref: %DerivedFromStringView.starts_with.cpp_overload_set.type = name_ref starts_with, imports.%DerivedFromStringView.starts_with.cpp_overload_set.value [concrete = constants.%DerivedFromStringView.starts_with.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.ref, %starts_with.ref
// CHECK:STDOUT:   %b.ref: %str = name_ref b, %b
// CHECK:STDOUT:   %.loc8_52: ref %str = value_as_ref %b.ref
// CHECK:STDOUT:   %addr.loc8_53.1: %ptr.85f = addr_of %.loc8_52
// CHECK:STDOUT:   %.loc8_53.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr.loc8_53.2: %ptr.bb2 = addr_of %.loc8_53.1
// CHECK:STDOUT:   %starts_with__carbon_thunk.call: init %empty_tuple.type = call imports.%starts_with__carbon_thunk.decl(%a.ref, %addr.loc8_53.1, %addr.loc8_53.2)
// CHECK:STDOUT:   %.loc8_53.2: init bool to %.loc8_53.1 = mark_in_place_init %starts_with__carbon_thunk.call
// CHECK:STDOUT:   return %.loc8_53.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

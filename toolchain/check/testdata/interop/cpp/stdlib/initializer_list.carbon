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
  let _: Cpp.std.initializer_list(i32) = (1, 2, 3);

  Cpp.Consume((1, 2, 3));

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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %initializer_list: type = class_type @initializer_list [concrete]
// CHECK:STDOUT:   %pattern_type.22f: type = pattern_type %initializer_list [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %.df2: Core.Form = init_form %initializer_list, call_param1 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.1: type = fn_type @initializer_list.initializer_list.loc8 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.1: %initializer_list.initializer_list.type.9fcb7f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.type: type = cpp_overload_set_type @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.82f: type = ptr_type %initializer_list [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk.type: type = fn_type @Consume__carbon_thunk [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk: %Consume__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.2: type = fn_type @initializer_list.initializer_list.loc10 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.2: %initializer_list.initializer_list.type.9fcb7f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor: type = class_type @InitListConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.92d: type = pattern_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %ptr.79d: type = ptr_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.type: type = fn_type @InitListConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk: %InitListConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.3: type = fn_type @initializer_list.initializer_list.loc12 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.3: %initializer_list.initializer_list.type.9fcb7f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.type: type = fn_type @InitListConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor: %InitListConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor.type: type = fn_type @initializer_list.cpp_destructor [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor: %initializer_list.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc12_44.3 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.1: %initializer_list.initializer_list.type.9fcb7f.1 = fn_decl @initializer_list.initializer_list.loc8 [concrete = constants.%initializer_list.initializer_list.643507.1] {
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern %_.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: Core.Form = init_form constants.%initializer_list, call_param1 [concrete = constants.%.df2]
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = value_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %Consume__carbon_thunk.decl: %Consume__carbon_thunk.type = fn_decl @Consume__carbon_thunk [concrete = constants.%Consume__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.2: %initializer_list.initializer_list.type.9fcb7f.2 = fn_decl @initializer_list.initializer_list.loc10 [concrete = constants.%initializer_list.initializer_list.643507.2] {
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern %_.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: Core.Form = init_form constants.%initializer_list, call_param1 [concrete = constants.%.df2]
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = value_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitListConstructor.decl: type = class_decl @InitListConstructor [concrete = constants.%InitListConstructor] {} {}
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.decl: %InitListConstructor__carbon_thunk.type = fn_decl @InitListConstructor__carbon_thunk [concrete = constants.%InitListConstructor__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.3: %initializer_list.initializer_list.type.9fcb7f.3 = fn_decl @initializer_list.initializer_list.loc12 [concrete = constants.%initializer_list.initializer_list.643507.3] {
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern %_.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12: Core.Form = init_form constants.%initializer_list, call_param1 [concrete = constants.%.df2]
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = value_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
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
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.22f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc8_43: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8_46: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc8_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc8_50.1: %tuple.type = tuple_literal (%int_1.loc8_43, %int_2.loc8_46, %int_3.loc8_49) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc8_50: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc8: type = array_type %int_3.loc8_50, %i32.1 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc8_50.2: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_50.1: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_50.1: <bound method> = bound_method %int_1.loc8_43, %impl.elem0.loc8_50.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc8_50.1: <specific function> = specific_function %impl.elem0.loc8_50.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.2: <bound method> = bound_method %int_1.loc8_43, %specific_fn.loc8_50.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.1: init %i32 = call %bound_method.loc8_50.2(%int_1.loc8_43) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_50.3: init %i32 = converted %int_1.loc8_43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_50.4: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc8: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc8_50.5: ref %i32 = array_index %.loc8_50.4, %int_0.loc8
// CHECK:STDOUT:   %.loc8_50.6: init %i32 to %.loc8_50.5 = in_place_init %.loc8_50.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc8_50.2: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_50.3: <bound method> = bound_method %int_2.loc8_46, %impl.elem0.loc8_50.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc8_50.2: <specific function> = specific_function %impl.elem0.loc8_50.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.4: <bound method> = bound_method %int_2.loc8_46, %specific_fn.loc8_50.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.2: init %i32 = call %bound_method.loc8_50.4(%int_2.loc8_46) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_50.7: init %i32 = converted %int_2.loc8_46, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc8_50: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_50.8: ref %i32 = array_index %.loc8_50.4, %int_1.loc8_50
// CHECK:STDOUT:   %.loc8_50.9: init %i32 to %.loc8_50.8 = in_place_init %.loc8_50.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc8_50.3: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_50.5: <bound method> = bound_method %int_3.loc8_49, %impl.elem0.loc8_50.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc8_50.3: <specific function> = specific_function %impl.elem0.loc8_50.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.6: <bound method> = bound_method %int_3.loc8_49, %specific_fn.loc8_50.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.3: init %i32 = call %bound_method.loc8_50.6(%int_3.loc8_49) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_50.10: init %i32 = converted %int_3.loc8_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc8_50: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc8_50.11: ref %i32 = array_index %.loc8_50.4, %int_2.loc8_50
// CHECK:STDOUT:   %.loc8_50.12: init %i32 to %.loc8_50.11 = in_place_init %.loc8_50.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_50.13: init %array_type to %.loc8_50.4 = array_init (%.loc8_50.6, %.loc8_50.9, %.loc8_50.12) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_50.14: init %array_type = converted %.loc8_50.1, %.loc8_50.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_50.15: ref %array_type = temporary %.loc8_50.4, %.loc8_50.14
// CHECK:STDOUT:   %.loc8_50.16: %array_type = acquire_value %.loc8_50.15
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc8: init %initializer_list to %.loc8_50.2 = call imports.%initializer_list.initializer_list.decl.2bdc3e.1(%.loc8_50.16)
// CHECK:STDOUT:   %.loc8_50.17: init %initializer_list = converted %.loc8_50.1, %initializer_list.initializer_list.call.loc8
// CHECK:STDOUT:   %.loc8_50.18: ref %initializer_list = temporary %.loc8_50.2, %.loc8_50.17
// CHECK:STDOUT:   %.loc8_50.19: %initializer_list = acquire_value %.loc8_50.18
// CHECK:STDOUT:   %_.loc8: %initializer_list = value_binding _, %.loc8_50.19
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %Consume.cpp_overload_set.type = name_ref Consume, imports.%Consume.cpp_overload_set.value [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc10_16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10_19: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc10_22: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc10_23.1: %tuple.type = tuple_literal (%int_1.loc10_16, %int_2.loc10_19, %int_3.loc10_22) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc10_23: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %int_3.loc10_23, %i32.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc10_23.2: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_23.1: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc10_23.1: <bound method> = bound_method %int_1.loc10_16, %impl.elem0.loc10_23.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc10_23.1: <specific function> = specific_function %impl.elem0.loc10_23.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.2: <bound method> = bound_method %int_1.loc10_16, %specific_fn.loc10_23.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.1: init %i32 = call %bound_method.loc10_23.2(%int_1.loc10_16) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.3: init %i32 = converted %int_1.loc10_16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_23.4: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc10: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_23.5: ref %i32 = array_index %.loc10_23.4, %int_0.loc10
// CHECK:STDOUT:   %.loc10_23.6: init %i32 to %.loc10_23.5 = in_place_init %.loc10_23.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_23.2: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc10_23.3: <bound method> = bound_method %int_2.loc10_19, %impl.elem0.loc10_23.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc10_23.2: <specific function> = specific_function %impl.elem0.loc10_23.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.4: <bound method> = bound_method %int_2.loc10_19, %specific_fn.loc10_23.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.2: init %i32 = call %bound_method.loc10_23.4(%int_2.loc10_19) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_23.7: init %i32 = converted %int_2.loc10_19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc10_23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_23.8: ref %i32 = array_index %.loc10_23.4, %int_1.loc10_23
// CHECK:STDOUT:   %.loc10_23.9: init %i32 to %.loc10_23.8 = in_place_init %.loc10_23.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc10_23.3: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc10_23.5: <bound method> = bound_method %int_3.loc10_22, %impl.elem0.loc10_23.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc10_23.3: <specific function> = specific_function %impl.elem0.loc10_23.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.6: <bound method> = bound_method %int_3.loc10_22, %specific_fn.loc10_23.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.3: init %i32 = call %bound_method.loc10_23.6(%int_3.loc10_22) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_23.10: init %i32 = converted %int_3.loc10_22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc10_23: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_23.11: ref %i32 = array_index %.loc10_23.4, %int_2.loc10_23
// CHECK:STDOUT:   %.loc10_23.12: init %i32 to %.loc10_23.11 = in_place_init %.loc10_23.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_23.13: init %array_type to %.loc10_23.4 = array_init (%.loc10_23.6, %.loc10_23.9, %.loc10_23.12) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_23.14: init %array_type = converted %.loc10_23.1, %.loc10_23.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_23.15: ref %array_type = temporary %.loc10_23.4, %.loc10_23.14
// CHECK:STDOUT:   %.loc10_23.16: %array_type = acquire_value %.loc10_23.15
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc10: init %initializer_list to %.loc10_23.2 = call imports.%initializer_list.initializer_list.decl.2bdc3e.2(%.loc10_23.16)
// CHECK:STDOUT:   %.loc10_23.17: init %initializer_list = converted %.loc10_23.1, %initializer_list.initializer_list.call.loc10
// CHECK:STDOUT:   %.loc10_23.18: ref %initializer_list = temporary %.loc10_23.2, %.loc10_23.17
// CHECK:STDOUT:   %.loc10_23.19: %initializer_list = acquire_value %.loc10_23.18
// CHECK:STDOUT:   %.loc10_23.20: ref %initializer_list = value_as_ref %.loc10_23.19
// CHECK:STDOUT:   %addr.loc10: %ptr.82f = addr_of %.loc10_23.20
// CHECK:STDOUT:   %Consume__carbon_thunk.call: init %empty_tuple.type = call imports.%Consume__carbon_thunk.decl(%addr.loc10)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.92d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc12_37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12_40: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc12_43: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_44.1: %tuple.type = tuple_literal (%int_1.loc12_37, %int_2.loc12_40, %int_3.loc12_43) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc12_13: type = splice_block %InitListConstructor.ref [concrete = constants.%InitListConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InitListConstructor.ref: type = name_ref InitListConstructor, imports.%InitListConstructor.decl [concrete = constants.%InitListConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_44.2: ref %InitListConstructor = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc12_44: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc12: type = array_type %int_3.loc12_44, %i32.3 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc12_44.3: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_44.1: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_44.1: <bound method> = bound_method %int_1.loc12_37, %impl.elem0.loc12_44.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc12_44.1: <specific function> = specific_function %impl.elem0.loc12_44.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.2: <bound method> = bound_method %int_1.loc12_37, %specific_fn.loc12_44.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.1: init %i32 = call %bound_method.loc12_44.2(%int_1.loc12_37) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_44.4: init %i32 = converted %int_1.loc12_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_44.5: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc12_44.6: ref %i32 = array_index %.loc12_44.5, %int_0.loc12
// CHECK:STDOUT:   %.loc12_44.7: init %i32 to %.loc12_44.6 = in_place_init %.loc12_44.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc12_44.2: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_44.3: <bound method> = bound_method %int_2.loc12_40, %impl.elem0.loc12_44.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc12_44.2: <specific function> = specific_function %impl.elem0.loc12_44.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.4: <bound method> = bound_method %int_2.loc12_40, %specific_fn.loc12_44.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.2: init %i32 = call %bound_method.loc12_44.4(%int_2.loc12_40) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_44.8: init %i32 = converted %int_2.loc12_40, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc12_44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_44.9: ref %i32 = array_index %.loc12_44.5, %int_1.loc12_44
// CHECK:STDOUT:   %.loc12_44.10: init %i32 to %.loc12_44.9 = in_place_init %.loc12_44.8 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc12_44.3: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_44.5: <bound method> = bound_method %int_3.loc12_43, %impl.elem0.loc12_44.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc12_44.3: <specific function> = specific_function %impl.elem0.loc12_44.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.6: <bound method> = bound_method %int_3.loc12_43, %specific_fn.loc12_44.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.3: init %i32 = call %bound_method.loc12_44.6(%int_3.loc12_43) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_44.11: init %i32 = converted %int_3.loc12_43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc12_44: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_44.12: ref %i32 = array_index %.loc12_44.5, %int_2.loc12_44
// CHECK:STDOUT:   %.loc12_44.13: init %i32 to %.loc12_44.12 = in_place_init %.loc12_44.11 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_44.14: init %array_type to %.loc12_44.5 = array_init (%.loc12_44.7, %.loc12_44.10, %.loc12_44.13) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc12_44.15: init %array_type = converted %.loc12_44.1, %.loc12_44.14 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc12_44.16: ref %array_type = temporary %.loc12_44.5, %.loc12_44.15
// CHECK:STDOUT:   %.loc12_44.17: %array_type = acquire_value %.loc12_44.16
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc12: init %initializer_list to %.loc12_44.3 = call imports.%initializer_list.initializer_list.decl.2bdc3e.3(%.loc12_44.17)
// CHECK:STDOUT:   %.loc12_44.18: init %initializer_list = converted %.loc12_44.1, %initializer_list.initializer_list.call.loc12
// CHECK:STDOUT:   %.loc12_44.19: ref %initializer_list = temporary %.loc12_44.3, %.loc12_44.18
// CHECK:STDOUT:   %.loc12_44.20: %initializer_list = acquire_value %.loc12_44.19
// CHECK:STDOUT:   %.loc12_44.21: ref %initializer_list = value_as_ref %.loc12_44.20
// CHECK:STDOUT:   %addr.loc12_44.1: %ptr.82f = addr_of %.loc12_44.21
// CHECK:STDOUT:   %addr.loc12_44.2: %ptr.79d = addr_of %.loc12_44.2
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InitListConstructor__carbon_thunk.decl(%addr.loc12_44.1, %addr.loc12_44.2)
// CHECK:STDOUT:   %.loc12_44.22: init %InitListConstructor to %.loc12_44.2 = mark_in_place_init %InitListConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_44.23: init %InitListConstructor = converted %.loc12_44.1, %.loc12_44.22
// CHECK:STDOUT:   %.loc12_44.24: ref %InitListConstructor = temporary %.loc12_44.2, %.loc12_44.23
// CHECK:STDOUT:   %.loc12_44.25: %InitListConstructor = acquire_value %.loc12_44.24
// CHECK:STDOUT:   %_.loc12: %InitListConstructor = value_binding _, %.loc12_44.25
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc12_44.24, constants.%InitListConstructor.cpp_destructor
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.call: init %empty_tuple.type = call %InitListConstructor.cpp_destructor.bound(%.loc12_44.24)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc12: <bound method> = bound_method %.loc12_44.19, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc12: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc12(%.loc12_44.19)
// CHECK:STDOUT:   %DestroyOp.bound.loc12: <bound method> = bound_method %.loc12_44.16, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc12: init %empty_tuple.type = call %DestroyOp.bound.loc12(%.loc12_44.16)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_23.18, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc10: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc10(%.loc10_23.18)
// CHECK:STDOUT:   %DestroyOp.bound.loc10: <bound method> = bound_method %.loc10_23.15, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10: init %empty_tuple.type = call %DestroyOp.bound.loc10(%.loc10_23.15)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_50.18, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc8: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc8(%.loc8_50.18)
// CHECK:STDOUT:   %DestroyOp.bound.loc8: <bound method> = bound_method %.loc8_50.15, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc8: init %empty_tuple.type = call %DestroyOp.bound.loc8(%.loc8_50.15)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc8(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc10(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc12(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc12_44.1(%self.param: ref %InitListConstructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc12_44.2(%self.param: ref %initializer_list) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc12_44.3(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.7fc: type = pattern_type %initializer_list [concrete]
// CHECK:STDOUT:   %_.patt.abb: %pattern_type.7fc = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %.0ee: Core.Form = init_form %initializer_list [concrete]
// CHECK:STDOUT:   %pattern_type.771: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %_.param_patt: %pattern_type.771 = value_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.03c: %pattern_type.771 = wrapper_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:   %return.param_patt.199: %pattern_type.7fc = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.07e: %pattern_type.7fc = return_slot_pattern %return.param_patt.199, %initializer_list [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.0fbcd7.1: type = fn_type @initializer_list.initializer_list.loc8 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.0db5f6.1: %initializer_list.initializer_list.type.0fbcd7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %.981: ref %array_type = temporary invalid, %array [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.type: type = cpp_overload_set_type @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Consume.cpp_overload_set.value: %Consume.cpp_overload_set.type = cpp_overload_set_value @Consume.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.fac: type = ptr_type %initializer_list [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk.type: type = fn_type @Consume__carbon_thunk [concrete]
// CHECK:STDOUT:   %Consume__carbon_thunk: %Consume__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.0fbcd7.2: type = fn_type @initializer_list.initializer_list.loc10 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.0db5f6.2: %initializer_list.initializer_list.type.0fbcd7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor.type: type = fn_type @initializer_list.cpp_destructor [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor: %initializer_list.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.type.facet: %type = facet_value %initializer_list, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.a8a: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%initializer_list.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.c44: %Destroy.WithSelf.SubobjectDestroy.type.a8a = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.782: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%initializer_list.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.589: %Destroy.WithSelf.SelfDestruct.type.782 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d01: <witness> = custom_witness (%initializer_list.cpp_destructor, %Destroy.WithSelf.SubobjectDestroy.c44, %Destroy.WithSelf.SelfDestruct.589), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.457: %Destroy.type = facet_value %initializer_list, (%custom_witness.d01) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc10_23.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.a6f: %type = facet_value %array_type, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.257: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%facet_value.a6f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.5e6: %Destroy.WithSelf.SubobjectDestroy.type.257 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.095: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%facet_value.a6f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.7bb: %Destroy.WithSelf.SelfDestruct.type.095 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e1e: <witness> = custom_witness (%Destroy.Op.1a2547.4, %Destroy.WithSelf.SubobjectDestroy.5e6, %Destroy.WithSelf.SelfDestruct.7bb), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.523: %Destroy.type = facet_value %array_type, (%custom_witness.e1e) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.977: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.7bb, @Destroy.WithSelf.SelfDestruct(%facet_value.a6f) [concrete]
// CHECK:STDOUT:   %bound_method.83e: <bound method> = bound_method %.981, %Destroy.WithSelf.SelfDestruct.specific_fn.977 [concrete]
// CHECK:STDOUT:   %InitListConstructor: type = class_type @InitListConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.ed6: type = pattern_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %_.patt.100: %pattern_type.ed6 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %ptr.cbb: type = ptr_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.type: type = fn_type @InitListConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk: %InitListConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.0fbcd7.3: type = fn_type @initializer_list.initializer_list.loc12 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.0db5f6.3: %initializer_list.initializer_list.type.0fbcd7.3 = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.type: type = fn_type @InitListConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor: %InitListConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor.type.facet: %type = facet_value %InitListConstructor, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.ffa: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%InitListConstructor.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.b24: %Destroy.WithSelf.SubobjectDestroy.type.ffa = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4b0: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%InitListConstructor.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.687: %Destroy.WithSelf.SelfDestruct.type.4b0 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.528: <witness> = custom_witness (%InitListConstructor.cpp_destructor, %Destroy.WithSelf.SubobjectDestroy.b24, %Destroy.WithSelf.SelfDestruct.687), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0b7: %Destroy.type = facet_value %InitListConstructor, (%custom_witness.528) [concrete]
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
// CHECK:STDOUT:   %initializer_list.decl: type = class_decl @initializer_list [concrete = constants.%initializer_list] {} {}
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.f8e97d.1: %initializer_list.initializer_list.type.0fbcd7.1 = fn_decl @initializer_list.initializer_list.loc8 [concrete = constants.%initializer_list.initializer_list.0db5f6.1] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.771 = value_param_pattern [concrete = constants.%_.param_patt]
// CHECK:STDOUT:     %_.patt: %pattern_type.771 = wrapper_binding_pattern _, %_.param_patt [concrete = constants.%_.patt.03c]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7fc = out_param_pattern [concrete = constants.%return.param_patt.199]
// CHECK:STDOUT:     %return.patt: %pattern_type.7fc = return_slot_pattern %return.param_patt, imports.%initializer_list.decl [concrete = constants.%return.patt.07e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = wrapper_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.f8e97d.2: %initializer_list.initializer_list.type.0fbcd7.2 = fn_decl @initializer_list.initializer_list.loc10 [concrete = constants.%initializer_list.initializer_list.0db5f6.2] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.771 = value_param_pattern [concrete = constants.%_.param_patt]
// CHECK:STDOUT:     %_.patt: %pattern_type.771 = wrapper_binding_pattern _, %_.param_patt [concrete = constants.%_.patt.03c]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7fc = out_param_pattern [concrete = constants.%return.param_patt.199]
// CHECK:STDOUT:     %return.patt: %pattern_type.7fc = return_slot_pattern %return.param_patt, imports.%initializer_list.decl [concrete = constants.%return.patt.07e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = wrapper_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.f8e97d.3: %initializer_list.initializer_list.type.0fbcd7.3 = fn_decl @initializer_list.initializer_list.loc12 [concrete = constants.%initializer_list.initializer_list.0db5f6.3] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.771 = value_param_pattern [concrete = constants.%_.param_patt]
// CHECK:STDOUT:     %_.patt: %pattern_type.771 = wrapper_binding_pattern _, %_.param_patt [concrete = constants.%_.patt.03c]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7fc = out_param_pattern [concrete = constants.%return.param_patt.199]
// CHECK:STDOUT:     %return.patt: %pattern_type.7fc = return_slot_pattern %return.param_patt, imports.%initializer_list.decl [concrete = constants.%return.patt.07e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = wrapper_binding _, %_.param
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
// CHECK:STDOUT:   %int_1.loc8_43: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8_46: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc8_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc8_50.1: %tuple.type = tuple_literal (%int_1.loc8_43, %int_2.loc8_46, %int_3.loc8_49) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc8_50: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc8: type = array_type %int_3.loc8_50, %i32.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc8_50.2: Core.Form = init_form imports.%initializer_list.decl [concrete = constants.%.0ee]
// CHECK:STDOUT:   %.loc8_50.3: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_50.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_50.1: <bound method> = bound_method %int_1.loc8_43, %impl.elem0.loc8_50.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc8_50.1: <specific function> = specific_function %impl.elem0.loc8_50.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.2: <bound method> = bound_method %int_1.loc8_43, %specific_fn.loc8_50.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.1: init %i32 = call %bound_method.loc8_50.2(%int_1.loc8_43) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc8_50.4: init %i32 = converted %int_1.loc8_43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc8_50.5: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc8: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc8_50.6: ref %i32 = array_index %.loc8_50.5, %int_0.loc8
// CHECK:STDOUT:   %.loc8_50.7: init %i32 to %.loc8_50.6 = in_place_init %.loc8_50.4 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc8_50.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_50.3: <bound method> = bound_method %int_2.loc8_46, %impl.elem0.loc8_50.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc8_50.2: <specific function> = specific_function %impl.elem0.loc8_50.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.4: <bound method> = bound_method %int_2.loc8_46, %specific_fn.loc8_50.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.2: init %i32 = call %bound_method.loc8_50.4(%int_2.loc8_46) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc8_50.8: init %i32 = converted %int_2.loc8_46, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc8_50: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_50.9: ref %i32 = array_index %.loc8_50.5, %int_1.loc8_50
// CHECK:STDOUT:   %.loc8_50.10: init %i32 to %.loc8_50.9 = in_place_init %.loc8_50.8 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc8_50.3: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_50.5: <bound method> = bound_method %int_3.loc8_49, %impl.elem0.loc8_50.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc8_50.3: <specific function> = specific_function %impl.elem0.loc8_50.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_50.6: <bound method> = bound_method %int_3.loc8_49, %specific_fn.loc8_50.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.3: init %i32 = call %bound_method.loc8_50.6(%int_3.loc8_49) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc8_50.11: init %i32 = converted %int_3.loc8_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_50.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc8_50: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc8_50.12: ref %i32 = array_index %.loc8_50.5, %int_2.loc8_50
// CHECK:STDOUT:   %.loc8_50.13: init %i32 to %.loc8_50.12 = in_place_init %.loc8_50.11 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc8_50.14: init %array_type to %.loc8_50.5 = array_init (%.loc8_50.7, %.loc8_50.10, %.loc8_50.13) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_50.15: init %array_type = converted %.loc8_50.1, %.loc8_50.14 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc8_50.16: ref %array_type = temporary %.loc8_50.5, %.loc8_50.15 [concrete = constants.%.981]
// CHECK:STDOUT:   %.loc8_50.17: %array_type = acquire_value %.loc8_50.16 [concrete = constants.%array]
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc8: init %initializer_list to %.loc8_50.3 = call generated.%initializer_list.initializer_list.decl.f8e97d.1(%.loc8_50.17)
// CHECK:STDOUT:   %.loc8_50.18: init %initializer_list = converted %.loc8_50.1, %initializer_list.initializer_list.call.loc8
// CHECK:STDOUT:   %.loc8_50.19: ref %initializer_list = temporary %.loc8_50.3, %.loc8_50.18
// CHECK:STDOUT:   %.loc8_50.20: %initializer_list = acquire_value %.loc8_50.19
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %_.loc8: %initializer_list = wrapper_binding _, %.loc8_50.20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc8: %pattern_type.7fc = value_binding_pattern _ [concrete = constants.%_.patt.abb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %Consume.cpp_overload_set.type = name_ref Consume, imports.%Consume.cpp_overload_set.value [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc10_16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10_19: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc10_22: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc10_23.1: %tuple.type = tuple_literal (%int_1.loc10_16, %int_2.loc10_19, %int_3.loc10_22) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc10_23: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %int_3.loc10_23, %i32.4 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc10_23.2: Core.Form = init_form imports.%initializer_list.decl [concrete = constants.%.0ee]
// CHECK:STDOUT:   %.loc10_23.3: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10_23.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_23.1: <bound method> = bound_method %int_1.loc10_16, %impl.elem0.loc10_23.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc10_23.1: <specific function> = specific_function %impl.elem0.loc10_23.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.2: <bound method> = bound_method %int_1.loc10_16, %specific_fn.loc10_23.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.1: init %i32 = call %bound_method.loc10_23.2(%int_1.loc10_16) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc10_23.4: init %i32 = converted %int_1.loc10_16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc10_23.5: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc10: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_23.6: ref %i32 = array_index %.loc10_23.5, %int_0.loc10
// CHECK:STDOUT:   %.loc10_23.7: init %i32 to %.loc10_23.6 = in_place_init %.loc10_23.4 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc10_23.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_23.3: <bound method> = bound_method %int_2.loc10_19, %impl.elem0.loc10_23.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc10_23.2: <specific function> = specific_function %impl.elem0.loc10_23.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.4: <bound method> = bound_method %int_2.loc10_19, %specific_fn.loc10_23.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.2: init %i32 = call %bound_method.loc10_23.4(%int_2.loc10_19) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc10_23.8: init %i32 = converted %int_2.loc10_19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc10_23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc10_23.9: ref %i32 = array_index %.loc10_23.5, %int_1.loc10_23
// CHECK:STDOUT:   %.loc10_23.10: init %i32 to %.loc10_23.9 = in_place_init %.loc10_23.8 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc10_23.3: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_23.5: <bound method> = bound_method %int_3.loc10_22, %impl.elem0.loc10_23.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc10_23.3: <specific function> = specific_function %impl.elem0.loc10_23.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_23.6: <bound method> = bound_method %int_3.loc10_22, %specific_fn.loc10_23.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.3: init %i32 = call %bound_method.loc10_23.6(%int_3.loc10_22) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc10_23.11: init %i32 = converted %int_3.loc10_22, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_23.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc10_23: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_23.12: ref %i32 = array_index %.loc10_23.5, %int_2.loc10_23
// CHECK:STDOUT:   %.loc10_23.13: init %i32 to %.loc10_23.12 = in_place_init %.loc10_23.11 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc10_23.14: init %array_type to %.loc10_23.5 = array_init (%.loc10_23.7, %.loc10_23.10, %.loc10_23.13) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_23.15: init %array_type = converted %.loc10_23.1, %.loc10_23.14 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_23.16: ref %array_type = temporary %.loc10_23.5, %.loc10_23.15 [concrete = constants.%.981]
// CHECK:STDOUT:   %.loc10_23.17: %array_type = acquire_value %.loc10_23.16 [concrete = constants.%array]
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc10: init %initializer_list to %.loc10_23.3 = call generated.%initializer_list.initializer_list.decl.f8e97d.2(%.loc10_23.17)
// CHECK:STDOUT:   %.loc10_23.18: init %initializer_list = converted %.loc10_23.1, %initializer_list.initializer_list.call.loc10
// CHECK:STDOUT:   %.loc10_23.19: ref %initializer_list = temporary %.loc10_23.3, %.loc10_23.18
// CHECK:STDOUT:   %.loc10_23.20: %initializer_list = acquire_value %.loc10_23.19
// CHECK:STDOUT:   %.loc10_23.21: ref %initializer_list = value_as_ref %.loc10_23.20
// CHECK:STDOUT:   %addr.loc10: %ptr.fac = addr_of %.loc10_23.21
// CHECK:STDOUT:   %Consume__carbon_thunk.call: init %empty_tuple.type = call imports.%Consume__carbon_thunk.decl(%addr.loc10)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %.loc10_23.19, constants.%Destroy.WithSelf.SelfDestruct.589
// CHECK:STDOUT:   %Destroy.facet.loc10_23.1: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc10_23.22: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc10_23.1 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %Destroy.facet.loc10_23.2: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc10_23.23: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc10_23.2 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_23.7: <bound method> = bound_method %.loc10_23.19, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_23.1: init %empty_tuple.type = call %bound_method.loc10_23.7(%.loc10_23.19)
// CHECK:STDOUT:   %Destroy.facet.loc10_23.3: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc10_23.24: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc10_23.3 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %Destroy.facet.loc10_23.4: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc10_23.25: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc10_23.4 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10_23.2: init %empty_tuple.type = call constants.%bound_method.83e(constants.%.981)
// CHECK:STDOUT:   %int_1.loc12_37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12_40: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc12_43: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_44.1: %tuple.type = tuple_literal (%int_1.loc12_37, %int_2.loc12_40, %int_3.loc12_43) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc12_44.2: ref %InitListConstructor = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc12_44: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc12: type = array_type %int_3.loc12_44, %i32.6 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc12_44.3: Core.Form = init_form imports.%initializer_list.decl [concrete = constants.%.0ee]
// CHECK:STDOUT:   %.loc12_44.4: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_44.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc12_44.1: <bound method> = bound_method %int_1.loc12_37, %impl.elem0.loc12_44.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc12_44.1: <specific function> = specific_function %impl.elem0.loc12_44.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.2: <bound method> = bound_method %int_1.loc12_37, %specific_fn.loc12_44.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.1: init %i32 = call %bound_method.loc12_44.2(%int_1.loc12_37) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc12_44.5: init %i32 = converted %int_1.loc12_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc12_44.6: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc12_44.7: ref %i32 = array_index %.loc12_44.6, %int_0.loc12
// CHECK:STDOUT:   %.loc12_44.8: init %i32 to %.loc12_44.7 = in_place_init %.loc12_44.5 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc12_44.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc12_44.3: <bound method> = bound_method %int_2.loc12_40, %impl.elem0.loc12_44.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc12_44.2: <specific function> = specific_function %impl.elem0.loc12_44.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.4: <bound method> = bound_method %int_2.loc12_40, %specific_fn.loc12_44.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.2: init %i32 = call %bound_method.loc12_44.4(%int_2.loc12_40) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc12_44.9: init %i32 = converted %int_2.loc12_40, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc12_44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_44.10: ref %i32 = array_index %.loc12_44.6, %int_1.loc12_44
// CHECK:STDOUT:   %.loc12_44.11: init %i32 to %.loc12_44.10 = in_place_init %.loc12_44.9 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc12_44.3: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc12_44.5: <bound method> = bound_method %int_3.loc12_43, %impl.elem0.loc12_44.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc12_44.3: <specific function> = specific_function %impl.elem0.loc12_44.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_44.6: <bound method> = bound_method %int_3.loc12_43, %specific_fn.loc12_44.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.3: init %i32 = call %bound_method.loc12_44.6(%int_3.loc12_43) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc12_44.12: init %i32 = converted %int_3.loc12_43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_44.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc12_44: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_44.13: ref %i32 = array_index %.loc12_44.6, %int_2.loc12_44
// CHECK:STDOUT:   %.loc12_44.14: init %i32 to %.loc12_44.13 = in_place_init %.loc12_44.12 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc12_44.15: init %array_type to %.loc12_44.6 = array_init (%.loc12_44.8, %.loc12_44.11, %.loc12_44.14) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc12_44.16: init %array_type = converted %.loc12_44.1, %.loc12_44.15 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc12_44.17: ref %array_type = temporary %.loc12_44.6, %.loc12_44.16 [concrete = constants.%.981]
// CHECK:STDOUT:   %.loc12_44.18: %array_type = acquire_value %.loc12_44.17 [concrete = constants.%array]
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc12: init %initializer_list to %.loc12_44.4 = call generated.%initializer_list.initializer_list.decl.f8e97d.3(%.loc12_44.18)
// CHECK:STDOUT:   %.loc12_44.19: init %initializer_list = converted %.loc12_44.1, %initializer_list.initializer_list.call.loc12
// CHECK:STDOUT:   %.loc12_44.20: ref %initializer_list = temporary %.loc12_44.4, %.loc12_44.19
// CHECK:STDOUT:   %.loc12_44.21: %initializer_list = acquire_value %.loc12_44.20
// CHECK:STDOUT:   %.loc12_44.22: ref %initializer_list = value_as_ref %.loc12_44.21
// CHECK:STDOUT:   %addr.loc12_44.1: %ptr.fac = addr_of %.loc12_44.22
// CHECK:STDOUT:   %addr.loc12_44.2: %ptr.cbb = addr_of %.loc12_44.2
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InitListConstructor__carbon_thunk.decl(%addr.loc12_44.1, %addr.loc12_44.2)
// CHECK:STDOUT:   %.loc12_44.23: init %InitListConstructor to %.loc12_44.2 = mark_in_place_init %InitListConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_44.24: init %InitListConstructor = converted %.loc12_44.1, %.loc12_44.23
// CHECK:STDOUT:   %.loc12_44.25: ref %InitListConstructor = temporary %.loc12_44.2, %.loc12_44.24
// CHECK:STDOUT:   %.loc12_44.26: %InitListConstructor = acquire_value %.loc12_44.25
// CHECK:STDOUT:   %.loc12_13: type = splice_block %InitListConstructor.ref [concrete = constants.%InitListConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InitListConstructor.ref: type = name_ref InitListConstructor, imports.%InitListConstructor.decl [concrete = constants.%InitListConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc12: %InitListConstructor = wrapper_binding _, %.loc12_44.26
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc12: %pattern_type.ed6 = value_binding_pattern _ [concrete = constants.%_.patt.100]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc12_44.1: <bound method> = bound_method %.loc12_44.25, constants.%Destroy.WithSelf.SelfDestruct.687
// CHECK:STDOUT:   %Destroy.facet.loc12_44.1: %Destroy.type = facet_value constants.%InitListConstructor, (constants.%custom_witness.528) [concrete = constants.%Destroy.facet.0b7]
// CHECK:STDOUT:   %.loc12_44.27: %Destroy.type = converted constants.%InitListConstructor, %Destroy.facet.loc12_44.1 [concrete = constants.%Destroy.facet.0b7]
// CHECK:STDOUT:   %Destroy.facet.loc12_44.2: %Destroy.type = facet_value constants.%InitListConstructor, (constants.%custom_witness.528) [concrete = constants.%Destroy.facet.0b7]
// CHECK:STDOUT:   %.loc12_44.28: %Destroy.type = converted constants.%InitListConstructor, %Destroy.facet.loc12_44.2 [concrete = constants.%Destroy.facet.0b7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_44.7: <bound method> = bound_method %.loc12_44.25, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc12_44.1: init %empty_tuple.type = call %bound_method.loc12_44.7(%.loc12_44.25)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc12_44.2: <bound method> = bound_method %.loc12_44.20, constants.%Destroy.WithSelf.SelfDestruct.589
// CHECK:STDOUT:   %Destroy.facet.loc12_44.3: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc12_44.29: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc12_44.3 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %Destroy.facet.loc12_44.4: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc12_44.30: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc12_44.4 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_44.8: <bound method> = bound_method %.loc12_44.20, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc12_44.2: init %empty_tuple.type = call %bound_method.loc12_44.8(%.loc12_44.20)
// CHECK:STDOUT:   %Destroy.facet.loc12_44.5: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc12_44.31: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc12_44.5 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %Destroy.facet.loc12_44.6: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc12_44.32: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc12_44.6 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc12_44.3: init %empty_tuple.type = call constants.%bound_method.83e(constants.%.981)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc8: <bound method> = bound_method %.loc8_50.19, constants.%Destroy.WithSelf.SelfDestruct.589
// CHECK:STDOUT:   %Destroy.facet.loc8_50.1: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc8_50.21: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc8_50.1 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %Destroy.facet.loc8_50.2: %Destroy.type = facet_value constants.%initializer_list, (constants.%custom_witness.d01) [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   %.loc8_50.22: %Destroy.type = converted constants.%initializer_list, %Destroy.facet.loc8_50.2 [concrete = constants.%Destroy.facet.457]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_50.7: <bound method> = bound_method %.loc8_50.19, %Destroy.WithSelf.SelfDestruct.specific_fn.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_50.1: init %empty_tuple.type = call %bound_method.loc8_50.7(%.loc8_50.19)
// CHECK:STDOUT:   %Destroy.facet.loc8_50.3: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc8_50.23: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc8_50.3 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %Destroy.facet.loc8_50.4: %Destroy.type = facet_value constants.%array_type, (constants.%custom_witness.e1e) [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   %.loc8_50.24: %Destroy.type = converted constants.%array_type, %Destroy.facet.loc8_50.4 [concrete = constants.%Destroy.facet.523]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8_50.2: init %empty_tuple.type = call constants.%bound_method.83e(constants.%.981)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc8(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc10(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_23.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_23.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_23.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc12(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %.b61: Core.Form = init_form %initializer_list [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.1: type = fn_type @initializer_list.initializer_list.loc32 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.1: %initializer_list.initializer_list.type.9fcb7f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.2: type = fn_type @initializer_list.initializer_list.loc44 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.2: %initializer_list.initializer_list.type.9fcb7f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor: type = class_type @InitListConstructor [concrete]
// CHECK:STDOUT:   %pattern_type.92d: type = pattern_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %ptr.79d: type = ptr_type %InitListConstructor [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.type: type = fn_type @InitListConstructor__carbon_thunk [concrete]
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk: %InitListConstructor__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.type.9fcb7f.3: type = fn_type @initializer_list.initializer_list.loc60 [concrete]
// CHECK:STDOUT:   %initializer_list.initializer_list.643507.3: %initializer_list.initializer_list.type.9fcb7f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.type: type = fn_type @InitListConstructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor: %InitListConstructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor.type: type = fn_type @initializer_list.cpp_destructor [concrete]
// CHECK:STDOUT:   %initializer_list.cpp_destructor: %initializer_list.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc60 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.1: %initializer_list.initializer_list.type.9fcb7f.1 = fn_decl @initializer_list.initializer_list.loc32 [concrete = constants.%initializer_list.initializer_list.643507.1] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = at_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern %return.param_patt, constants.%initializer_list [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32: Core.Form = init_form constants.%initializer_list [concrete = constants.%.b61]
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = value_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.2: %initializer_list.initializer_list.type.9fcb7f.2 = fn_decl @initializer_list.initializer_list.loc44 [concrete = constants.%initializer_list.initializer_list.643507.2] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = at_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern %return.param_patt, constants.%initializer_list [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc44: Core.Form = init_form constants.%initializer_list [concrete = constants.%.b61]
// CHECK:STDOUT:     %_.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %_: %array_type = value_binding _, %_.param
// CHECK:STDOUT:     %return.param: ref %initializer_list = out_param call_param1
// CHECK:STDOUT:     %return: ref %initializer_list = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %initializer_list.initializer_list.decl.2bdc3e.3: %initializer_list.initializer_list.type.9fcb7f.3 = fn_decl @initializer_list.initializer_list.loc60 [concrete = constants.%initializer_list.initializer_list.643507.3] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.5d8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %_.patt: %pattern_type.5d8 = at_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.22f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.22f = return_slot_pattern %return.param_patt, constants.%initializer_list [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc60: Core.Form = init_form constants.%initializer_list [concrete = constants.%.b61]
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
// CHECK:STDOUT:     %_.patt.loc15: %pattern_type.22f = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc29: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc31: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc32.1: %tuple.type = tuple_literal (%int_1.loc27, %int_2.loc29, %int_3.loc31) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc32: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc32: type = array_type %int_3.loc32, %i32.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc32.2: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc32.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc32.1: <bound method> = bound_method %int_1.loc27, %impl.elem0.loc32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc32.1: <specific function> = specific_function %impl.elem0.loc32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32.2: <bound method> = bound_method %int_1.loc27, %specific_fn.loc32.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.1: init %i32 = call %bound_method.loc32.2(%int_1.loc27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.3: init %i32 = converted %int_1.loc27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.4: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc32.5: ref %i32 = array_index %.loc32.4, %int_0.loc32
// CHECK:STDOUT:   %.loc32.6: init %i32 to %.loc32.5 = in_place_init %.loc32.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc32.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc32.3: <bound method> = bound_method %int_2.loc29, %impl.elem0.loc32.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc32.2: <specific function> = specific_function %impl.elem0.loc32.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32.4: <bound method> = bound_method %int_2.loc29, %specific_fn.loc32.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.2: init %i32 = call %bound_method.loc32.4(%int_2.loc29) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc32.7: init %i32 = converted %int_2.loc29, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc32.8: ref %i32 = array_index %.loc32.4, %int_1.loc32
// CHECK:STDOUT:   %.loc32.9: init %i32 to %.loc32.8 = in_place_init %.loc32.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc32.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc32.5: <bound method> = bound_method %int_3.loc31, %impl.elem0.loc32.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc32.3: <specific function> = specific_function %impl.elem0.loc32.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32.6: <bound method> = bound_method %int_3.loc31, %specific_fn.loc32.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.3: init %i32 = call %bound_method.loc32.6(%int_3.loc31) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc32.10: init %i32 = converted %int_3.loc31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc32: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc32.11: ref %i32 = array_index %.loc32.4, %int_2.loc32
// CHECK:STDOUT:   %.loc32.12: init %i32 to %.loc32.11 = in_place_init %.loc32.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc32.13: init %array_type to %.loc32.4 = array_init (%.loc32.6, %.loc32.9, %.loc32.12) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc32.14: init %array_type = converted %.loc32.1, %.loc32.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc32.15: ref %array_type = temporary %.loc32.4, %.loc32.14
// CHECK:STDOUT:   %.loc32.16: %array_type = acquire_value %.loc32.15
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc32: init %initializer_list to %.loc32.2 = call generated.%initializer_list.initializer_list.decl.2bdc3e.1(%.loc32.16)
// CHECK:STDOUT:   %.loc32.17: init %initializer_list = converted %.loc32.1, %initializer_list.initializer_list.call.loc32
// CHECK:STDOUT:   %.loc32.18: ref %initializer_list = temporary %.loc32.2, %.loc32.17
// CHECK:STDOUT:   %.loc32.19: %initializer_list = acquire_value %.loc32.18
// CHECK:STDOUT:   %_.loc15: %initializer_list = value_binding _, %.loc32.19
// CHECK:STDOUT:   %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Consume.ref: %Consume.cpp_overload_set.type = name_ref Consume, imports.%Consume.cpp_overload_set.value [concrete = constants.%Consume.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc39: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc41: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc43: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc44.1: %tuple.type = tuple_literal (%int_1.loc39, %int_2.loc41, %int_3.loc43) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc44: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc44: type = array_type %int_3.loc44, %i32.4 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc44.2: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc44.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.1: <bound method> = bound_method %int_1.loc39, %impl.elem0.loc44.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc44.1: <specific function> = specific_function %impl.elem0.loc44.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.2: <bound method> = bound_method %int_1.loc39, %specific_fn.loc44.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.1: init %i32 = call %bound_method.loc44.2(%int_1.loc39) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc44.3: init %i32 = converted %int_1.loc39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc44.4: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc44: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc44.5: ref %i32 = array_index %.loc44.4, %int_0.loc44
// CHECK:STDOUT:   %.loc44.6: init %i32 to %.loc44.5 = in_place_init %.loc44.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc44.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.3: <bound method> = bound_method %int_2.loc41, %impl.elem0.loc44.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc44.2: <specific function> = specific_function %impl.elem0.loc44.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.4: <bound method> = bound_method %int_2.loc41, %specific_fn.loc44.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.2: init %i32 = call %bound_method.loc44.4(%int_2.loc41) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc44.7: init %i32 = converted %int_2.loc41, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc44.8: ref %i32 = array_index %.loc44.4, %int_1.loc44
// CHECK:STDOUT:   %.loc44.9: init %i32 to %.loc44.8 = in_place_init %.loc44.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc44.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.5: <bound method> = bound_method %int_3.loc43, %impl.elem0.loc44.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc44.3: <specific function> = specific_function %impl.elem0.loc44.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.6: <bound method> = bound_method %int_3.loc43, %specific_fn.loc44.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.3: init %i32 = call %bound_method.loc44.6(%int_3.loc43) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc44.10: init %i32 = converted %int_3.loc43, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc44.11: ref %i32 = array_index %.loc44.4, %int_2.loc44
// CHECK:STDOUT:   %.loc44.12: init %i32 to %.loc44.11 = in_place_init %.loc44.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc44.13: init %array_type to %.loc44.4 = array_init (%.loc44.6, %.loc44.9, %.loc44.12) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc44.14: init %array_type = converted %.loc44.1, %.loc44.13 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc44.15: ref %array_type = temporary %.loc44.4, %.loc44.14
// CHECK:STDOUT:   %.loc44.16: %array_type = acquire_value %.loc44.15
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc44: init %initializer_list to %.loc44.2 = call generated.%initializer_list.initializer_list.decl.2bdc3e.2(%.loc44.16)
// CHECK:STDOUT:   %.loc44.17: init %initializer_list = converted %.loc44.1, %initializer_list.initializer_list.call.loc44
// CHECK:STDOUT:   %.loc44.18: ref %initializer_list = temporary %.loc44.2, %.loc44.17
// CHECK:STDOUT:   %.loc44.19: %initializer_list = acquire_value %.loc44.18
// CHECK:STDOUT:   %.loc44.20: ref %initializer_list = value_as_ref %.loc44.19
// CHECK:STDOUT:   %addr.loc45: %ptr.82f = addr_of %.loc44.20
// CHECK:STDOUT:   %Consume__carbon_thunk.call: init %empty_tuple.type = call imports.%Consume__carbon_thunk.decl(%addr.loc45)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc48: %pattern_type.92d = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.loc55: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc57: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc59: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc60.1: %tuple.type = tuple_literal (%int_1.loc55, %int_2.loc57, %int_3.loc59) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc51: type = splice_block %InitListConstructor.ref [concrete = constants.%InitListConstructor] {
// CHECK:STDOUT:     %Cpp.ref.loc50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %InitListConstructor.ref: type = name_ref InitListConstructor, imports.%InitListConstructor.decl [concrete = constants.%InitListConstructor]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc60.2: ref %InitListConstructor = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_3.loc60: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %array_type.loc60: type = array_type %int_3.loc60, %i32.6 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc60.3: ref %initializer_list = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc60.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc60.1: <bound method> = bound_method %int_1.loc55, %impl.elem0.loc60.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc60.1: <specific function> = specific_function %impl.elem0.loc60.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60.2: <bound method> = bound_method %int_1.loc55, %specific_fn.loc60.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.1: init %i32 = call %bound_method.loc60.2(%int_1.loc55) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc60.4: init %i32 = converted %int_1.loc55, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc60.5: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0.loc60: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc60.6: ref %i32 = array_index %.loc60.5, %int_0.loc60
// CHECK:STDOUT:   %.loc60.7: init %i32 to %.loc60.6 = in_place_init %.loc60.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc60.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc60.3: <bound method> = bound_method %int_2.loc57, %impl.elem0.loc60.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc60.2: <specific function> = specific_function %impl.elem0.loc60.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60.4: <bound method> = bound_method %int_2.loc57, %specific_fn.loc60.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.2: init %i32 = call %bound_method.loc60.4(%int_2.loc57) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc60.8: init %i32 = converted %int_2.loc57, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc60: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc60.9: ref %i32 = array_index %.loc60.5, %int_1.loc60
// CHECK:STDOUT:   %.loc60.10: init %i32 to %.loc60.9 = in_place_init %.loc60.8 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc60.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc60.5: <bound method> = bound_method %int_3.loc59, %impl.elem0.loc60.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc60.3: <specific function> = specific_function %impl.elem0.loc60.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60.6: <bound method> = bound_method %int_3.loc59, %specific_fn.loc60.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.3: init %i32 = call %bound_method.loc60.6(%int_3.loc59) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc60.11: init %i32 = converted %int_3.loc59, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc60: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc60.12: ref %i32 = array_index %.loc60.5, %int_2.loc60
// CHECK:STDOUT:   %.loc60.13: init %i32 to %.loc60.12 = in_place_init %.loc60.11 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc60.14: init %array_type to %.loc60.5 = array_init (%.loc60.7, %.loc60.10, %.loc60.13) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc60.15: init %array_type = converted %.loc60.1, %.loc60.14 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc60.16: ref %array_type = temporary %.loc60.5, %.loc60.15
// CHECK:STDOUT:   %.loc60.17: %array_type = acquire_value %.loc60.16
// CHECK:STDOUT:   %initializer_list.initializer_list.call.loc60: init %initializer_list to %.loc60.3 = call generated.%initializer_list.initializer_list.decl.2bdc3e.3(%.loc60.17)
// CHECK:STDOUT:   %.loc60.18: init %initializer_list = converted %.loc60.1, %initializer_list.initializer_list.call.loc60
// CHECK:STDOUT:   %.loc60.19: ref %initializer_list = temporary %.loc60.3, %.loc60.18
// CHECK:STDOUT:   %.loc60.20: %initializer_list = acquire_value %.loc60.19
// CHECK:STDOUT:   %.loc60.21: ref %initializer_list = value_as_ref %.loc60.20
// CHECK:STDOUT:   %addr.loc60.1: %ptr.82f = addr_of %.loc60.21
// CHECK:STDOUT:   %addr.loc60.2: %ptr.79d = addr_of %.loc60.2
// CHECK:STDOUT:   %InitListConstructor__carbon_thunk.call: init %empty_tuple.type = call imports.%InitListConstructor__carbon_thunk.decl(%addr.loc60.1, %addr.loc60.2)
// CHECK:STDOUT:   %.loc60.22: init %InitListConstructor to %.loc60.2 = mark_in_place_init %InitListConstructor__carbon_thunk.call
// CHECK:STDOUT:   %.loc60.23: init %InitListConstructor = converted %.loc60.1, %.loc60.22
// CHECK:STDOUT:   %.loc60.24: ref %InitListConstructor = temporary %.loc60.2, %.loc60.23
// CHECK:STDOUT:   %.loc60.25: %InitListConstructor = acquire_value %.loc60.24
// CHECK:STDOUT:   %_.loc48: %InitListConstructor = value_binding _, %.loc60.25
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.bound: <bound method> = bound_method %.loc60.24, constants.%InitListConstructor.cpp_destructor
// CHECK:STDOUT:   %InitListConstructor.cpp_destructor.call: init %empty_tuple.type = call %InitListConstructor.cpp_destructor.bound(%.loc60.24)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc60: <bound method> = bound_method %.loc60.19, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc60: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc60(%.loc60.19)
// CHECK:STDOUT:   %Destroy.Op.bound.loc60: <bound method> = bound_method %.loc60.16, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc60: init %empty_tuple.type = call %Destroy.Op.bound.loc60(%.loc60.16)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc44: <bound method> = bound_method %.loc44.18, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc44: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc44(%.loc44.18)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44: <bound method> = bound_method %.loc44.15, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44: init %empty_tuple.type = call %Destroy.Op.bound.loc44(%.loc44.15)
// CHECK:STDOUT:   %initializer_list.cpp_destructor.bound.loc32: <bound method> = bound_method %.loc32.18, constants.%initializer_list.cpp_destructor
// CHECK:STDOUT:   %initializer_list.cpp_destructor.call.loc32: init %empty_tuple.type = call %initializer_list.cpp_destructor.bound.loc32(%.loc32.18)
// CHECK:STDOUT:   %Destroy.Op.bound.loc32: <bound method> = bound_method %.loc32.15, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc32: init %empty_tuple.type = call %Destroy.Op.bound.loc32(%.loc32.15)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc32(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc44(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @initializer_list.initializer_list.loc60(%_.param: %array_type) -> out %return.param: %initializer_list = "cpp.std.initializer_list.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc60(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:

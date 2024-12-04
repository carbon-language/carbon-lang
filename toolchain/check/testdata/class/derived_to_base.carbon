// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/derived_to_base.carbon

base class A {
  var a: i32;
}

base class B {
  extend base: A;
  var b: i32;
}

class C {
  extend base: B;
  var c: i32;
}

fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
}

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.a.1: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.a.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem.1: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %B.elem.2: type = unbound_element_type %B, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.b.1: type = struct_type {.base: %A, .b: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.b.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %C.elem.2: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %struct_type.base.c.1: type = struct_type {.base: %B, .c: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.c.1 [template]
// CHECK:STDOUT:   %ptr.4: type = ptr_type %C [template]
// CHECK:STDOUT:   %ptr.5: type = ptr_type %B [template]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [template]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.8: type = ptr_type %A [template]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [template]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [template]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [template]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [template]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [template]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [template]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.a.2: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.base.b.3: type = struct_type {.base: %struct_type.a.2, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %struct_type.base.c.3: type = struct_type {.base: %struct_type.base.b.3, .c: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.2) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %int_2.2) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %int_3.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ConvertCToB = %ConvertCToB.decl
// CHECK:STDOUT:     .ConvertBToA = %ConvertBToA.decl
// CHECK:STDOUT:     .ConvertCToA = %ConvertCToA.decl
// CHECK:STDOUT:     .ConvertValue = %ConvertValue.decl
// CHECK:STDOUT:     .ConvertRef = %ConvertRef.decl
// CHECK:STDOUT:     .ConvertInit = %ConvertInit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [template = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.patt: %ptr.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc25_20: type = ptr_type %C [template = constants.%ptr.4]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %ptr.loc25_27: type = ptr_type %B [template = constants.%ptr.5]
// CHECK:STDOUT:     %p.param: %ptr.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [template = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %ptr.5 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.5 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.8 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %ptr.loc26_20: type = ptr_type %B [template = constants.%ptr.5]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %ptr.loc26_27: type = ptr_type %A [template = constants.%ptr.8]
// CHECK:STDOUT:     %p.param: %ptr.5 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.5 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.8 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [template = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %ptr.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.8 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc27_20: type = ptr_type %C [template = constants.%ptr.4]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %ptr.loc27_27: type = ptr_type %A [template = constants.%ptr.8]
// CHECK:STDOUT:     %p.param: %ptr.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.8 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [template = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [template = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.patt: %ptr.4 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %ptr.4 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.8 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc33_19: type = ptr_type %C [template = constants.%ptr.4]
// CHECK:STDOUT:     %A.ref.loc33: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %ptr.loc33_26: type = ptr_type %A [template = constants.%ptr.8]
// CHECK:STDOUT:     %c.param: %ptr.4 = value_param runtime_param0
// CHECK:STDOUT:     %c: %ptr.4 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.8 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [template = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %A.elem = field_decl a, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.1 [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %B.elem.1 = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.2: type = converted %int.make_type_signed, %.loc17_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8: %B.elem.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .b = %.loc17_8
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %C.elem.1 = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.2: type = converted %int.make_type_signed, %.loc22_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8: %C.elem.2 = field_decl c, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.c.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .c = %.loc22_8
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param_patt: %ptr.4) -> %ptr.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc25_39.2: ref %B = class_element_access %.loc25_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.5 = addr_of %.loc25_39.2
// CHECK:STDOUT:   %.loc25_39.3: %ptr.5 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc25_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param_patt: %ptr.5) -> %ptr.8 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.5 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc26_39.2: ref %A = class_element_access %.loc26_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.8 = addr_of %.loc26_39.2
// CHECK:STDOUT:   %.loc26_39.3: %ptr.8 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc26_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param_patt: %ptr.4) -> %ptr.8 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc27_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc27_39.2: ref %B = class_element_access %.loc27_39.1, element0
// CHECK:STDOUT:   %.loc27_39.3: ref %A = class_element_access %.loc27_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.8 = addr_of %.loc27_39.3
// CHECK:STDOUT:   %.loc27_39.4: %ptr.8 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc27_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc30_15.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc30_15.2: ref %A = class_element_access %.loc30_15.1, element0
// CHECK:STDOUT:   %.loc30_15.3: ref %A = converted %c.ref, %.loc30_15.2
// CHECK:STDOUT:   %.loc30_15.4: %A = bind_value %.loc30_15.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc30_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param_patt: %ptr.4) -> %ptr.8 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.4 = name_ref c, %c
// CHECK:STDOUT:   %.loc34_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc34: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc34_15.1: ref %B = class_element_access %.loc34_12, element0
// CHECK:STDOUT:   %.loc34_15.2: ref %A = class_element_access %.loc34_15.1, element0
// CHECK:STDOUT:   %.loc34_15.3: ref %A = converted %.loc34_12, %.loc34_15.2
// CHECK:STDOUT:   %addr: %ptr.8 = addr_of %.loc34_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc38_39.1: %struct_type.a.2 = struct_literal (%int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc38_48.1: %struct_type.base.b.3 = struct_literal (%.loc38_39.1, %int_2)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc38_57.1: %struct_type.base.c.3 = struct_literal (%.loc38_48.1, %int_3)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc38_39: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc38_39: <bound method> = bound_method %int_1, %impl.elem0.loc38_39 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc38_39: <specific function> = specific_function %Convert.bound.loc38_39, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc38_39: init %i32 = call %Convert.specific_fn.loc38_39(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc38_39.2: init %i32 = converted %int_1, %int.convert_checked.loc38_39 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc38_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_57.3: ref %B = class_element_access %.loc38_57.2, element0
// CHECK:STDOUT:   %.loc38_48.2: ref %A = class_element_access %.loc38_57.3, element0
// CHECK:STDOUT:   %.loc38_39.3: ref %i32 = class_element_access %.loc38_48.2, element0
// CHECK:STDOUT:   %.loc38_39.4: init %i32 = initialize_from %.loc38_39.2 to %.loc38_39.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc38_39.5: init %A = class_init (%.loc38_39.4), %.loc38_48.2 [template = constants.%A.val]
// CHECK:STDOUT:   %.loc38_48.3: init %A = converted %.loc38_39.1, %.loc38_39.5 [template = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc38_48: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc38_48: <bound method> = bound_method %int_2, %impl.elem0.loc38_48 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc38_48: <specific function> = specific_function %Convert.bound.loc38_48, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc38_48: init %i32 = call %Convert.specific_fn.loc38_48(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc38_48.4: init %i32 = converted %int_2, %int.convert_checked.loc38_48 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc38_48.5: ref %i32 = class_element_access %.loc38_57.3, element1
// CHECK:STDOUT:   %.loc38_48.6: init %i32 = initialize_from %.loc38_48.4 to %.loc38_48.5 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc38_48.7: init %B = class_init (%.loc38_48.3, %.loc38_48.6), %.loc38_57.3 [template = constants.%B.val]
// CHECK:STDOUT:   %.loc38_57.4: init %B = converted %.loc38_48.1, %.loc38_48.7 [template = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc38_57: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc38_57: <bound method> = bound_method %int_3, %impl.elem0.loc38_57 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc38_57: <specific function> = specific_function %Convert.bound.loc38_57, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc38_57: init %i32 = call %Convert.specific_fn.loc38_57(%int_3) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc38_57.5: init %i32 = converted %int_3, %int.convert_checked.loc38_57 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc38_57.6: ref %i32 = class_element_access %.loc38_57.2, element1
// CHECK:STDOUT:   %.loc38_57.7: init %i32 = initialize_from %.loc38_57.5 to %.loc38_57.6 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc38_57.8: init %C = class_init (%.loc38_57.4, %.loc38_57.7), %.loc38_57.2 [template = constants.%C.val]
// CHECK:STDOUT:   %.loc38_57.9: ref %C = temporary %.loc38_57.2, %.loc38_57.8
// CHECK:STDOUT:   %.loc38_59: ref %C = converted %.loc38_57.1, %.loc38_57.9
// CHECK:STDOUT:   %.loc38_63.1: ref %B = class_element_access %.loc38_59, element0
// CHECK:STDOUT:   %.loc38_63.2: ref %A = class_element_access %.loc38_63.1, element0
// CHECK:STDOUT:   %.loc38_63.3: ref %A = converted %.loc38_59, %.loc38_63.2
// CHECK:STDOUT:   %.loc38_63.4: %A = bind_value %.loc38_63.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc38_63.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

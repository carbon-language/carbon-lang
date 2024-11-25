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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %B, %i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %A, .b: %i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.13: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.14: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: %B, .c: %i32} [template]
// CHECK:STDOUT:   %.16: <witness> = complete_type_witness %.15 [template]
// CHECK:STDOUT:   %.17: type = ptr_type %C [template]
// CHECK:STDOUT:   %.18: type = ptr_type %B [template]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [template]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [template]
// CHECK:STDOUT:   %.22: type = ptr_type %A [template]
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
// CHECK:STDOUT:   %.23: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.24: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.25: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.26: type = struct_type {.base: %.24, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.28: type = struct_type {.base: %.26, .c: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.52: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.53: <bound method> = bound_method %.23, %Convert.14 [template]
// CHECK:STDOUT:   %.54: <specific function> = specific_function %.53, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.55: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.1: %A = struct_value (%.55) [template]
// CHECK:STDOUT:   %.56: <bound method> = bound_method %.25, %Convert.14 [template]
// CHECK:STDOUT:   %.57: <specific function> = specific_function %.56, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.58: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct.2: %B = struct_value (%struct.1, %.58) [template]
// CHECK:STDOUT:   %.59: <bound method> = bound_method %.27, %Convert.14 [template]
// CHECK:STDOUT:   %.60: <specific function> = specific_function %.59, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.61: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct.3: %C = struct_value (%struct.2, %.61) [template]
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
// CHECK:STDOUT:     %p.patt: %.17 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.17 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.18 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.18 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc25_20: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc25_27: type = ptr_type %B [template = constants.%.18]
// CHECK:STDOUT:     %p.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.17 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.18 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.18 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [template = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %.18 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.18 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc26_20: type = ptr_type %B [template = constants.%.18]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc26_27: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %p.param: %.18 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.18 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [template = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %.17 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.17 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc27_20: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc27_27: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %p.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.17 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
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
// CHECK:STDOUT:     %c.patt: %.17 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %.17 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc33_19: type = ptr_type %C [template = constants.%.17]
// CHECK:STDOUT:     %A.ref.loc33: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc33_26: type = ptr_type %A [template = constants.%.22]
// CHECK:STDOUT:     %c.param: %.17 = value_param runtime_param0
// CHECK:STDOUT:     %c: %.17 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [template = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.6 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %.loc17_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc17_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_10.3: type = converted %int.make_type_signed, %.loc17_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8: %.7 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .b = %.loc17_8
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %.13 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %.loc22_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc22_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.3: type = converted %int.make_type_signed, %.loc22_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8: %.14 = field_decl c, element1 [template]
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.15 [template = constants.%.16]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .c = %.loc22_8
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param_patt: %.17) -> %.18 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.17 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc25_39.2: ref %B = class_element_access %.loc25_39.1, element0
// CHECK:STDOUT:   %.loc25_39.3: %.18 = addr_of %.loc25_39.2
// CHECK:STDOUT:   %.loc25_39.4: %.18 = converted %p.ref, %.loc25_39.3
// CHECK:STDOUT:   return %.loc25_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param_patt: %.18) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.18 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc26_39.2: ref %A = class_element_access %.loc26_39.1, element0
// CHECK:STDOUT:   %.loc26_39.3: %.22 = addr_of %.loc26_39.2
// CHECK:STDOUT:   %.loc26_39.4: %.22 = converted %p.ref, %.loc26_39.3
// CHECK:STDOUT:   return %.loc26_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param_patt: %.17) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.17 = name_ref p, %p
// CHECK:STDOUT:   %.loc27_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc27_39.2: ref %B = class_element_access %.loc27_39.1, element0
// CHECK:STDOUT:   %.loc27_39.3: ref %A = class_element_access %.loc27_39.2, element0
// CHECK:STDOUT:   %.loc27_39.4: %.22 = addr_of %.loc27_39.3
// CHECK:STDOUT:   %.loc27_39.5: %.22 = converted %p.ref, %.loc27_39.4
// CHECK:STDOUT:   return %.loc27_39.5
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
// CHECK:STDOUT: fn @ConvertRef(%c.param_patt: %.17) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %.17 = name_ref c, %c
// CHECK:STDOUT:   %.loc34_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc34: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc34_15.1: ref %B = class_element_access %.loc34_12, element0
// CHECK:STDOUT:   %.loc34_15.2: ref %A = class_element_access %.loc34_15.1, element0
// CHECK:STDOUT:   %.loc34_15.3: ref %A = converted %.loc34_12, %.loc34_15.2
// CHECK:STDOUT:   %.loc34_10: %.22 = addr_of %.loc34_15.3
// CHECK:STDOUT:   return %.loc34_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc38_38: Core.IntLiteral = int_value 1 [template = constants.%.23]
// CHECK:STDOUT:   %.loc38_39.1: %.24 = struct_literal (%.loc38_38)
// CHECK:STDOUT:   %.loc38_47: Core.IntLiteral = int_value 2 [template = constants.%.25]
// CHECK:STDOUT:   %.loc38_48.1: %.26 = struct_literal (%.loc38_39.1, %.loc38_47)
// CHECK:STDOUT:   %.loc38_56: Core.IntLiteral = int_value 3 [template = constants.%.27]
// CHECK:STDOUT:   %.loc38_57.1: %.28 = struct_literal (%.loc38_48.1, %.loc38_56)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc38_39.2: %Convert.type.2 = interface_witness_access constants.%.52, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc38_39.3: <bound method> = bound_method %.loc38_38, %.loc38_39.2 [template = constants.%.53]
// CHECK:STDOUT:   %.loc38_39.4: <specific function> = specific_function %.loc38_39.3, @Convert.2(constants.%.1) [template = constants.%.54]
// CHECK:STDOUT:   %int.convert_checked.loc38_39: init %i32 = call %.loc38_39.4(%.loc38_38) [template = constants.%.55]
// CHECK:STDOUT:   %.loc38_39.5: init %i32 = converted %.loc38_38, %int.convert_checked.loc38_39 [template = constants.%.55]
// CHECK:STDOUT:   %.loc38_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_57.3: ref %B = class_element_access %.loc38_57.2, element0
// CHECK:STDOUT:   %.loc38_48.2: ref %A = class_element_access %.loc38_57.3, element0
// CHECK:STDOUT:   %.loc38_39.6: ref %i32 = class_element_access %.loc38_48.2, element0
// CHECK:STDOUT:   %.loc38_39.7: init %i32 = initialize_from %.loc38_39.5 to %.loc38_39.6 [template = constants.%.55]
// CHECK:STDOUT:   %.loc38_39.8: init %A = class_init (%.loc38_39.7), %.loc38_48.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc38_48.3: init %A = converted %.loc38_39.1, %.loc38_39.8 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc38_48.4: %Convert.type.2 = interface_witness_access constants.%.52, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc38_48.5: <bound method> = bound_method %.loc38_47, %.loc38_48.4 [template = constants.%.56]
// CHECK:STDOUT:   %.loc38_48.6: <specific function> = specific_function %.loc38_48.5, @Convert.2(constants.%.1) [template = constants.%.57]
// CHECK:STDOUT:   %int.convert_checked.loc38_48: init %i32 = call %.loc38_48.6(%.loc38_47) [template = constants.%.58]
// CHECK:STDOUT:   %.loc38_48.7: init %i32 = converted %.loc38_47, %int.convert_checked.loc38_48 [template = constants.%.58]
// CHECK:STDOUT:   %.loc38_48.8: ref %i32 = class_element_access %.loc38_57.3, element1
// CHECK:STDOUT:   %.loc38_48.9: init %i32 = initialize_from %.loc38_48.7 to %.loc38_48.8 [template = constants.%.58]
// CHECK:STDOUT:   %.loc38_48.10: init %B = class_init (%.loc38_48.3, %.loc38_48.9), %.loc38_57.3 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc38_57.4: init %B = converted %.loc38_48.1, %.loc38_48.10 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc38_57.5: %Convert.type.2 = interface_witness_access constants.%.52, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc38_57.6: <bound method> = bound_method %.loc38_56, %.loc38_57.5 [template = constants.%.59]
// CHECK:STDOUT:   %.loc38_57.7: <specific function> = specific_function %.loc38_57.6, @Convert.2(constants.%.1) [template = constants.%.60]
// CHECK:STDOUT:   %int.convert_checked.loc38_57: init %i32 = call %.loc38_57.7(%.loc38_56) [template = constants.%.61]
// CHECK:STDOUT:   %.loc38_57.8: init %i32 = converted %.loc38_56, %int.convert_checked.loc38_57 [template = constants.%.61]
// CHECK:STDOUT:   %.loc38_57.9: ref %i32 = class_element_access %.loc38_57.2, element1
// CHECK:STDOUT:   %.loc38_57.10: init %i32 = initialize_from %.loc38_57.8 to %.loc38_57.9 [template = constants.%.61]
// CHECK:STDOUT:   %.loc38_57.11: init %C = class_init (%.loc38_57.4, %.loc38_57.10), %.loc38_57.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc38_57.12: ref %C = temporary %.loc38_57.2, %.loc38_57.11
// CHECK:STDOUT:   %.loc38_59: ref %C = converted %.loc38_57.1, %.loc38_57.12
// CHECK:STDOUT:   %.loc38_63.1: ref %B = class_element_access %.loc38_59, element0
// CHECK:STDOUT:   %.loc38_63.2: ref %A = class_element_access %.loc38_63.1, element0
// CHECK:STDOUT:   %.loc38_63.3: ref %A = converted %.loc38_59, %.loc38_63.2
// CHECK:STDOUT:   %.loc38_63.4: %A = bind_value %.loc38_63.3
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc38_63.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_field.carbon

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn Access(p: Derived*) -> i32* {
  return &(*p).c;
}

// CHECK:STDOUT: --- base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, %i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %.11: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %p.patt: %.10 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.10 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.11 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.11 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc24_21: type = ptr_type %Derived [template = constants.%.10]
// CHECK:STDOUT:     %.loc24_27: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc24_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_30.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_30.2: type = converted %int.make_type_signed, %.loc24_30.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_30.3: type = ptr_type %i32 [template = constants.%.11]
// CHECK:STDOUT:     %p.param: %.10 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.10 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.11 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.11 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.3: type = converted %int.make_type_signed.loc13, %.loc13_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc14_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.2: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.3: type = converted %int.make_type_signed.loc14, %.loc14_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8: %.2 = field_decl c, element2 [template]
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT:   .c = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc18: %.6 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %.loc20_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_10.2: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_10.3: type = converted %int.make_type_signed.loc20, %.loc20_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc21_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.2: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.3: type = converted %int.make_type_signed.loc21, %.loc21_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_8: %.7 = field_decl e, element2 [template]
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .d = %.loc20_8
// CHECK:STDOUT:   .e = %.loc21_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p.param_patt: %.10) -> %.11 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.10 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_12: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: %.2 = name_ref c, @Base.%.loc14_8 [template = @Base.%.loc14_8]
// CHECK:STDOUT:   %.loc25_15.1: ref %Base = class_element_access %.loc25_12, element0
// CHECK:STDOUT:   %.loc25_15.2: ref %Base = converted %.loc25_12, %.loc25_15.1
// CHECK:STDOUT:   %.loc25_15.3: ref %i32 = class_element_access %.loc25_15.2, element2
// CHECK:STDOUT:   %.loc25_10: %.11 = addr_of %.loc25_15.3
// CHECK:STDOUT:   return %.loc25_10
// CHECK:STDOUT: }
// CHECK:STDOUT:

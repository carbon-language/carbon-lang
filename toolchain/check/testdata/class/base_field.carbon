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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Base, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %Base, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %.10: type = ptr_type i32 [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
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
// CHECK:STDOUT:     %p.patt: %.9 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.9 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.10 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.10 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc24_21: type = ptr_type %Derived [template = constants.%.9]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc24_30.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc24_30.2: type = converted %int.make_type_32, %.loc24_30.1 [template = i32]
// CHECK:STDOUT:     %.loc24_30.3: type = ptr_type i32 [template = constants.%.10]
// CHECK:STDOUT:     %p.param: %.9 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.9 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.10 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.10 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.1 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:   %.loc13_8: %.1 = field_decl b, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_10.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_10.2: type = converted %int.make_type_32.loc14, %.loc14_10.1 [template = i32]
// CHECK:STDOUT:   %.loc14_8: %.1 = field_decl c, element2 [template]
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.2 [template = constants.%.3]
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
// CHECK:STDOUT:   %.loc18: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_10.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_10.2: type = converted %int.make_type_32.loc20, %.loc20_10.1 [template = i32]
// CHECK:STDOUT:   %.loc20_8: %.6 = field_decl d, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_32.loc21, %.loc21_10.1 [template = i32]
// CHECK:STDOUT:   %.loc21_8: %.6 = field_decl e, element2 [template]
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.7 [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .d = %.loc20_8
// CHECK:STDOUT:   .e = %.loc21_8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p.param_patt: %.9) -> %.10 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_12: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: %.1 = name_ref c, @Base.%.loc14_8 [template = @Base.%.loc14_8]
// CHECK:STDOUT:   %.loc25_15.1: ref %Base = class_element_access %.loc25_12, element0
// CHECK:STDOUT:   %.loc25_15.2: ref %Base = converted %.loc25_12, %.loc25_15.1
// CHECK:STDOUT:   %.loc25_15.3: ref i32 = class_element_access %.loc25_15.2, element2
// CHECK:STDOUT:   %.loc25_10: %.10 = addr_of %.loc25_15.3
// CHECK:STDOUT:   return %.loc25_10
// CHECK:STDOUT: }
// CHECK:STDOUT:

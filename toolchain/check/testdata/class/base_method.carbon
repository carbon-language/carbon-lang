// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method.carbon

base class Base {
  var a: i32;

  fn F[addr self: Self*]();
}

fn Base.F[addr self: Self*]() {
  (*self).a = 1;
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  (*p).F();
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Base, %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: %.2} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.34: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.35: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.36: <witness> = complete_type_witness %.35 [template]
// CHECK:STDOUT:   %.37: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %.4 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.4 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17_11: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc17_26: type = ptr_type %Base [template = constants.%.4]
// CHECK:STDOUT:     %self.param.loc17: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc17: %.4 = bind_name self, %self.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %.37 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.37 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc25: type = ptr_type %Derived [template = constants.%.37]
// CHECK:STDOUT:     %p.param: %.37 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.37 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc12_10.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed, %.loc12_10.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_8: %.3 = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %.4 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.4 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17_11: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc14: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc14: type = ptr_type %Base [template = constants.%.4]
// CHECK:STDOUT:     %self.param.loc14: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc14: %.4 = bind_name self, %self.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc22: %.34 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.35 [template = constants.%.36]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt: %.4]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.4 = name_ref self, %self.loc17
// CHECK:STDOUT:   %.loc18_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %.3 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc18_10: ref %.2 = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %.loc18_15: Core.IntLiteral = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc18_13.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_13.2: <bound method> = bound_method %.loc18_15, %.loc18_13.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_13.3: <specific function> = specific_function %.loc18_13.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc18_13.3(%.loc18_15) [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_13.4: init %.2 = converted %.loc18_15, %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   assign %.loc18_10, %.loc18_13.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param_patt: %.37) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.37 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_4.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Base.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc26_7: <bound method> = bound_method %.loc26_4.1, %F.ref
// CHECK:STDOUT:   %.loc26_4.2: %.37 = addr_of %.loc26_4.1
// CHECK:STDOUT:   %.loc26_4.3: ref %Derived = deref %.loc26_4.2
// CHECK:STDOUT:   %.loc26_4.4: ref %Base = class_element_access %.loc26_4.3, element0
// CHECK:STDOUT:   %.loc26_4.5: %.4 = addr_of %.loc26_4.4
// CHECK:STDOUT:   %.loc26_4.6: %.4 = converted %.loc26_4.2, %.loc26_4.5
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc26_7(%.loc26_4.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

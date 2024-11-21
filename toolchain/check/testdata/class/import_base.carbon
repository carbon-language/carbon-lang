// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_base.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn F[self: Self]();
  fn Unused[self: Self]();

  var x: i32;
  var unused: i32;
}

class Child {
  extend base: Base;
}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Child = {.base = {.x = 0, .unused = 1}};
  a.x = 2;
  a.F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Unused.type: type = fn_type @Unused [template]
// CHECK:STDOUT:   %Unused: %Unused.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %i32, .unused: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Child, %Base [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
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
// CHECK:STDOUT:     .Child = %Child.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Child.decl: type = class_decl @Child [template = constants.%Child] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.param: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unused.decl: %Unused.type = fn_decl @Unused [template = constants.%Unused] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.param: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.3: type = converted %int.make_type_signed.loc8, %.loc8_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc9_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_15.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_15.3: type = converted %int.make_type_signed.loc9, %.loc9_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_13: %.2 = field_decl unused, element1 [template]
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Unused = %Unused.decl
// CHECK:STDOUT:   .x = %.loc8_8
// CHECK:STDOUT:   .unused = %.loc9_13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: %.6 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.7 [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Child
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unused[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.11: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.x: Core.IntLiteral, .unused: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: %.12} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.10, %Convert.14 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.11, %Convert.14 [template]
// CHECK:STDOUT:   %.42: <specific function> = specific_function %.41, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.43: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.1: %Base = struct_value (%.40, %.43) [template]
// CHECK:STDOUT:   %struct.2: %Child = struct_value (%struct.1) [template]
// CHECK:STDOUT:   %.44: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.14 [template]
// CHECK:STDOUT:   %.47: <specific function> = specific_function %.46, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.48: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+49, loaded [template = constants.%Child]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.11
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//a, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.4: %F.type = import_ref Main//a, inst+10, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.6: %.44 = import_ref Main//a, inst+40, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//a, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//a, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//a, inst+54, unloaded
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//a, inst+51, loaded [template = constants.%Base]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = imports.%import_ref.1
// CHECK:STDOUT:     .Child = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   .base = imports.%import_ref.9
// CHECK:STDOUT:   extend imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .F = imports.%import_ref.4
// CHECK:STDOUT:   .Unused = imports.%import_ref.5
// CHECK:STDOUT:   .x = imports.%import_ref.6
// CHECK:STDOUT:   .unused = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, imports.%import_ref.2 [template = constants.%Child]
// CHECK:STDOUT:   %a.var: ref %Child = var a
// CHECK:STDOUT:   %a: ref %Child = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_33: Core.IntLiteral = int_value 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_46: Core.IntLiteral = int_value 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_47.1: %.12 = struct_literal (%.loc7_33, %.loc7_46)
// CHECK:STDOUT:   %.loc7_48.1: %.13 = struct_literal (%.loc7_47.1)
// CHECK:STDOUT:   %.loc7_47.2: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_47.3: <bound method> = bound_method %.loc7_33, %.loc7_47.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc7_47.4: <specific function> = specific_function %.loc7_47.3, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.1: init %i32 = call %.loc7_47.4(%.loc7_33) [template = constants.%.40]
// CHECK:STDOUT:   %.loc7_47.5: init %i32 = converted %.loc7_33, %int.convert_checked.loc7_47.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc7_48.2: ref %Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_47.6: ref %i32 = class_element_access %.loc7_48.2, element0
// CHECK:STDOUT:   %.loc7_47.7: init %i32 = initialize_from %.loc7_47.5 to %.loc7_47.6 [template = constants.%.40]
// CHECK:STDOUT:   %.loc7_47.8: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_47.9: <bound method> = bound_method %.loc7_46, %.loc7_47.8 [template = constants.%.41]
// CHECK:STDOUT:   %.loc7_47.10: <specific function> = specific_function %.loc7_47.9, @Convert.2(constants.%.1) [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc7_47.2: init %i32 = call %.loc7_47.10(%.loc7_46) [template = constants.%.43]
// CHECK:STDOUT:   %.loc7_47.11: init %i32 = converted %.loc7_46, %int.convert_checked.loc7_47.2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc7_47.12: ref %i32 = class_element_access %.loc7_48.2, element1
// CHECK:STDOUT:   %.loc7_47.13: init %i32 = initialize_from %.loc7_47.11 to %.loc7_47.12 [template = constants.%.43]
// CHECK:STDOUT:   %.loc7_47.14: init %Base = class_init (%.loc7_47.7, %.loc7_47.13), %.loc7_48.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc7_48.3: init %Base = converted %.loc7_47.1, %.loc7_47.14 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc7_48.4: init %Child = class_init (%.loc7_48.3), %a.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_49: init %Child = converted %.loc7_48.1, %.loc7_48.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign %a.var, %.loc7_49
// CHECK:STDOUT:   %a.ref.loc8: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: %.44 = name_ref x, imports.%import_ref.6 [template = imports.%.1]
// CHECK:STDOUT:   %.loc8_4.1: ref %Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref %Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref %i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %.loc8_9: Core.IntLiteral = int_value 2 [template = constants.%.45]
// CHECK:STDOUT:   %.loc8_7.1: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_7.2: <bound method> = bound_method %.loc8_9, %.loc8_7.1 [template = constants.%.46]
// CHECK:STDOUT:   %.loc8_7.3: <specific function> = specific_function %.loc8_7.2, @Convert.2(constants.%.1) [template = constants.%.47]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %.loc8_7.3(%.loc8_9) [template = constants.%.48]
// CHECK:STDOUT:   %.loc8_7.4: init %i32 = converted %.loc8_9, %int.convert_checked.loc8 [template = constants.%.48]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_7.4
// CHECK:STDOUT:   %a.ref.loc9: ref %Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.4 [template = constants.%F]
// CHECK:STDOUT:   %.loc9_4: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_3.1: ref %Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.2: ref %Base = converted %a.ref.loc9, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.3: %Base = bind_value %.loc9_3.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc9_4(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %Base]();
// CHECK:STDOUT:

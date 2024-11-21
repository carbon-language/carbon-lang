// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Empty {
}

class Field {
  var x: i32;
}

class ForwardDeclared;

class ForwardDeclared {
  fn F[self: Self]();
  fn G[addr self: Self*]();
}

class Incomplete;

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn Run() {
  var a: Empty = {};

  var b: Field = {.x = 1};
  b.x = 2;

  var c: ForwardDeclared = {};
  c.F();
  c.G();

  var d: ForwardDeclared* = &c;

  var e: Incomplete*;
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Field: type = class_type @Field [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Field, %i32 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %ForwardDeclared: type = class_type @ForwardDeclared [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type %ForwardDeclared [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
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
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Field = %Field.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc11
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = class_decl @Empty [template = constants.%Empty] {} {}
// CHECK:STDOUT:   %Field.decl: type = class_decl @Field [template = constants.%Field] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc11: type = class_decl @ForwardDeclared [template = constants.%ForwardDeclared] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc13: type = class_decl @ForwardDeclared [template = constants.%ForwardDeclared] {} {}
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Empty
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field {
// CHECK:STDOUT:   %.loc8_10.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc8_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.3: type = converted %int.make_type_signed, %.loc8_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8: %.4 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Field
// CHECK:STDOUT:   .x = %.loc8_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %ForwardDeclared = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ForwardDeclared = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDeclared [template = constants.%ForwardDeclared]
// CHECK:STDOUT:     %self.param: %ForwardDeclared = value_param runtime_param0
// CHECK:STDOUT:     %self: %ForwardDeclared = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %.7 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.7 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDeclared [template = constants.%ForwardDeclared]
// CHECK:STDOUT:     %.loc15_23: type = ptr_type %ForwardDeclared [template = constants.%.7]
// CHECK:STDOUT:     %self.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.7 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardDeclared
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %ForwardDeclared]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self.param_patt: %.7]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %struct.1: %Empty = struct_value () [template]
// CHECK:STDOUT:   %Field: type = class_type @Field [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.4) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.4) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.2: %Field = struct_value (%.36) [template]
// CHECK:STDOUT:   %.37: type = unbound_element_type %Field, %i32 [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.38, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.4) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %ForwardDeclared.1: type = class_type @ForwardDeclared.1 [template]
// CHECK:STDOUT:   %struct.3: %ForwardDeclared.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.42: type = ptr_type %ForwardDeclared.1 [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.43: type = ptr_type %Incomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+3, loaded [template = constants.%Empty]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+8, loaded [template = constants.%Field]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//a, inst+33, loaded [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//a, inst+56, loaded [template = constants.%Incomplete]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//a, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.7: %.37 = import_ref Main//a, inst+29, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.44 = import_ref Main//a, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.45: %F.type = import_ref Main//a, inst+41, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.46: %G.type = import_ref Main//a, inst+52, loaded [template = constants.%G]
// CHECK:STDOUT:   %import_ref.47 = import_ref Main//a, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.48 = import_ref Main//a, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.49 = import_ref Main//a, inst+52, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = imports.%import_ref.1
// CHECK:STDOUT:     .Field = imports.%import_ref.2
// CHECK:STDOUT:     .ForwardDeclared = imports.%import_ref.3
// CHECK:STDOUT:     .Incomplete = imports.%import_ref.4
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   .x = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.44
// CHECK:STDOUT:   .F = imports.%import_ref.45
// CHECK:STDOUT:   .G = imports.%import_ref.46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.47
// CHECK:STDOUT:   .F = imports.%import_ref.48
// CHECK:STDOUT:   .G = imports.%import_ref.49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Empty.ref: type = name_ref Empty, imports.%import_ref.1 [template = constants.%Empty]
// CHECK:STDOUT:   %a.var: ref %Empty = var a
// CHECK:STDOUT:   %a: ref %Empty = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_19.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc7_19.2: init %Empty = class_init (), %a.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc7_20: init %Empty = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %a.var, %.loc7_20
// CHECK:STDOUT:   %Field.ref: type = name_ref Field, imports.%import_ref.2 [template = constants.%Field]
// CHECK:STDOUT:   %b.var: ref %Field = var b
// CHECK:STDOUT:   %b: ref %Field = bind_name b, %b.var
// CHECK:STDOUT:   %.loc9_24: Core.IntLiteral = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_25.1: %.9 = struct_literal (%.loc9_24)
// CHECK:STDOUT:   %.loc9_25.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_25.3: <bound method> = bound_method %.loc9_24, %.loc9_25.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_25.4: <specific function> = specific_function %.loc9_25.3, @Convert.2(constants.%.4) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %.loc9_25.4(%.loc9_24) [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_25.5: init %i32 = converted %.loc9_24, %int.convert_checked.loc9 [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_25.6: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc9_25.7: init %i32 = initialize_from %.loc9_25.5 to %.loc9_25.6 [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_25.8: init %Field = class_init (%.loc9_25.7), %b.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_26: init %Field = converted %.loc9_25.1, %.loc9_25.8 [template = constants.%struct.2]
// CHECK:STDOUT:   assign %b.var, %.loc9_26
// CHECK:STDOUT:   %b.ref: ref %Field = name_ref b, %b
// CHECK:STDOUT:   %x.ref: %.37 = name_ref x, imports.%import_ref.7 [template = imports.%.1]
// CHECK:STDOUT:   %.loc10_4: ref %i32 = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc10_9: Core.IntLiteral = int_value 2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc10_7.1: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_7.2: <bound method> = bound_method %.loc10_9, %.loc10_7.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc10_7.3: <specific function> = specific_function %.loc10_7.2, @Convert.2(constants.%.4) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %.loc10_7.3(%.loc10_9) [template = constants.%.41]
// CHECK:STDOUT:   %.loc10_7.4: init %i32 = converted %.loc10_9, %int.convert_checked.loc10 [template = constants.%.41]
// CHECK:STDOUT:   assign %.loc10_4, %.loc10_7.4
// CHECK:STDOUT:   %ForwardDeclared.ref.loc12: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %c.var: ref %ForwardDeclared.1 = var c
// CHECK:STDOUT:   %c: ref %ForwardDeclared.1 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc12_29.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc12_29.2: init %ForwardDeclared.1 = class_init (), %c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc12_30: init %ForwardDeclared.1 = converted %.loc12_29.1, %.loc12_29.2 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %c.var, %.loc12_30
// CHECK:STDOUT:   %c.ref.loc13: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.45 [template = constants.%F]
// CHECK:STDOUT:   %.loc13_4: <bound method> = bound_method %c.ref.loc13, %F.ref
// CHECK:STDOUT:   %.loc13_3: %ForwardDeclared.1 = bind_value %c.ref.loc13
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc13_4(%.loc13_3)
// CHECK:STDOUT:   %c.ref.loc14: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%import_ref.46 [template = constants.%G]
// CHECK:STDOUT:   %.loc14_4: <bound method> = bound_method %c.ref.loc14, %G.ref
// CHECK:STDOUT:   %.loc14_3: %.42 = addr_of %c.ref.loc14
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %.loc14_4(%.loc14_3)
// CHECK:STDOUT:   %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %.loc16_25: type = ptr_type %ForwardDeclared.1 [template = constants.%.42]
// CHECK:STDOUT:   %d.var: ref %.42 = var d
// CHECK:STDOUT:   %d: ref %.42 = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref.loc16: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc16_29: %.42 = addr_of %c.ref.loc16
// CHECK:STDOUT:   assign %d.var, %.loc16_29
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, imports.%import_ref.4 [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc18: type = ptr_type %Incomplete [template = constants.%.43]
// CHECK:STDOUT:   %e.var: ref %.43 = var e
// CHECK:STDOUT:   %e: ref %.43 = bind_name e, %e.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %ForwardDeclared.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr <unexpected>.inst+443: %.42]();
// CHECK:STDOUT:

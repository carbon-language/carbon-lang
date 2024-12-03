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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Field: type = class_type @Field [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Field, %i32 [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %ForwardDeclared: type = class_type @ForwardDeclared [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %ForwardDeclared [template]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Empty
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_signed, %.loc8_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8: %.1 = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Field
// CHECK:STDOUT:   .x = %.loc8_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:     %self.patt: %.2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.2 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc15_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDeclared [template = constants.%ForwardDeclared]
// CHECK:STDOUT:     %.loc15_23: type = ptr_type %ForwardDeclared [template = constants.%.2]
// CHECK:STDOUT:     %self.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardDeclared
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %ForwardDeclared]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self.param_patt: %.2]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %struct.1: %Empty = struct_value () [template]
// CHECK:STDOUT:   %Field: type = class_type @Field [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.1 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct.2: %Field = struct_value (%int.1.2) [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Field, %i32 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %ForwardDeclared.1: type = class_type @ForwardDeclared.1 [template]
// CHECK:STDOUT:   %struct.3: %ForwardDeclared.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type %ForwardDeclared.1 [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.9: type = ptr_type %Incomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+3, loaded [template = constants.%Empty]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+8, loaded [template = constants.%Field]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//a, inst+31, loaded [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %import_ref.4: type = import_ref Main//a, inst+54, loaded [template = constants.%Incomplete]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.10
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: <witness> = import_ref Main//a, inst+6, loaded [template = constants.%complete_type.1]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//a, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.7: <witness> = import_ref Main//a, inst+29, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//a, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.9: %.7 = import_ref Main//a, inst+27, loaded [template = %.1]
// CHECK:STDOUT:   %import_ref.46: <witness> = import_ref Main//a, inst+53, loaded [template = constants.%complete_type.1]
// CHECK:STDOUT:   %import_ref.47 = import_ref Main//a, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.48: %F.type = import_ref Main//a, inst+39, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.49: %G.type = import_ref Main//a, inst+50, loaded [template = constants.%G]
// CHECK:STDOUT:   %import_ref.50: <witness> = import_ref Main//a, inst+53, loaded [template = constants.%complete_type.1]
// CHECK:STDOUT:   %import_ref.51 = import_ref Main//a, inst+32, unloaded
// CHECK:STDOUT:   %import_ref.52 = import_ref Main//a, inst+39, unloaded
// CHECK:STDOUT:   %import_ref.53 = import_ref Main//a, inst+50, unloaded
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
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Field {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   .x = imports.%import_ref.9
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.47
// CHECK:STDOUT:   .F = imports.%import_ref.48
// CHECK:STDOUT:   .G = imports.%import_ref.49
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDeclared.2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.51
// CHECK:STDOUT:   .F = imports.%import_ref.52
// CHECK:STDOUT:   .G = imports.%import_ref.53
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Empty.ref: type = name_ref Empty, imports.%import_ref.1 [template = constants.%Empty]
// CHECK:STDOUT:   %a.var: ref %Empty = var a
// CHECK:STDOUT:   %a: ref %Empty = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_19.2: init %Empty = class_init (), %a.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc7_20: init %Empty = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %a.var, %.loc7_20
// CHECK:STDOUT:   %Field.ref: type = name_ref Field, imports.%import_ref.2 [template = constants.%Field]
// CHECK:STDOUT:   %b.var: ref %Field = var b
// CHECK:STDOUT:   %b: ref %Field = bind_name b, %b.var
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc9_25.1: %struct_type.x.2 = struct_literal (%int.1)
// CHECK:STDOUT:   %.loc9_25.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9: <bound method> = bound_method %int.1, %.loc9_25.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9: <specific function> = specific_function %Convert.bound.loc9, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %Convert.specific_fn.loc9(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_25.3: init %i32 = converted %int.1, %int.convert_checked.loc9 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_25.4: ref %i32 = class_element_access %b.var, element0
// CHECK:STDOUT:   %.loc9_25.5: init %i32 = initialize_from %.loc9_25.3 to %.loc9_25.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_25.6: init %Field = class_init (%.loc9_25.5), %b.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_26: init %Field = converted %.loc9_25.1, %.loc9_25.6 [template = constants.%struct.2]
// CHECK:STDOUT:   assign %b.var, %.loc9_26
// CHECK:STDOUT:   %b.ref: ref %Field = name_ref b, %b
// CHECK:STDOUT:   %x.ref: %.7 = name_ref x, imports.%import_ref.9 [template = imports.%.1]
// CHECK:STDOUT:   %.loc10_4: ref %i32 = class_element_access %b.ref, element0
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc10_7.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int.2, %.loc10_7.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc10_7.2: init %i32 = converted %int.2, %int.convert_checked.loc10 [template = constants.%int.2.2]
// CHECK:STDOUT:   assign %.loc10_4, %.loc10_7.2
// CHECK:STDOUT:   %ForwardDeclared.ref.loc12: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %c.var: ref %ForwardDeclared.1 = var c
// CHECK:STDOUT:   %c: ref %ForwardDeclared.1 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc12_29.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_29.2: init %ForwardDeclared.1 = class_init (), %c.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc12_30: init %ForwardDeclared.1 = converted %.loc12_29.1, %.loc12_29.2 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %c.var, %.loc12_30
// CHECK:STDOUT:   %c.ref.loc13: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.48 [template = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %c.ref.loc13, %F.ref
// CHECK:STDOUT:   %.loc13: %ForwardDeclared.1 = bind_value %c.ref.loc13
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc13)
// CHECK:STDOUT:   %c.ref.loc14: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%import_ref.49 [template = constants.%G]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %c.ref.loc14, %G.ref
// CHECK:STDOUT:   %.loc14: %.8 = addr_of %c.ref.loc14
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.bound(%.loc14)
// CHECK:STDOUT:   %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.1]
// CHECK:STDOUT:   %.loc16_25: type = ptr_type %ForwardDeclared.1 [template = constants.%.8]
// CHECK:STDOUT:   %d.var: ref %.8 = var d
// CHECK:STDOUT:   %d: ref %.8 = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref.loc16: ref %ForwardDeclared.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc16_29: %.8 = addr_of %c.ref.loc16
// CHECK:STDOUT:   assign %d.var, %.loc16_29
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, imports.%import_ref.4 [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc18: type = ptr_type %Incomplete [template = constants.%.9]
// CHECK:STDOUT:   %e.var: ref %.9 = var e
// CHECK:STDOUT:   %e: ref %.9 = bind_name e, %e.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %ForwardDeclared.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr <unexpected>.inst+403: %.8]();
// CHECK:STDOUT:

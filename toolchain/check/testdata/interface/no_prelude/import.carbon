// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/import.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

interface Empty {
}

interface Basic {
  let T:! type;
  fn F();
}

// TODO: Implement merging of interface on import.
// interface ForwardDeclared;

interface ForwardDeclared {
  let T:! type;
  fn F();
}

var f_ref: {.f: ForwardDeclared};

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

fn UseEmpty(e: Empty) {}
fn UseBasic(e: Basic) {}
fn UseForwardDeclared(f: ForwardDeclared) {}

alias UseBasicT = Basic.T;
alias UseBasicF = Basic.F;

alias UseForwardDeclaredT = ForwardDeclared.T;
alias UseForwardDeclaredF = ForwardDeclared.F;

var f: ForwardDeclared* = &f_ref.f;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self.1: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Basic.type: type = interface_type @Basic [template]
// CHECK:STDOUT:   %Self.2: %Basic.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Basic.type, type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Basic.%T [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Basic.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, @Basic.%F.decl [template]
// CHECK:STDOUT:   %ForwardDeclared.type: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %Self.3: %ForwardDeclared.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ForwardDeclared.type, type [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @ForwardDeclared.%T [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ForwardDeclared.type, %F.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element1, @ForwardDeclared.%F.decl [template]
// CHECK:STDOUT:   %.10: type = struct_type {.f: %ForwardDeclared.type} [template]
// CHECK:STDOUT:   %.11: type = struct_type {.f: %.3} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Basic = %Basic.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl
// CHECK:STDOUT:     .f_ref = %f_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %Basic.decl: type = interface_decl @Basic [template = constants.%Basic.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl: type = interface_decl @ForwardDeclared [template = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %.loc20: type = struct_type {.f: %ForwardDeclared.type} [template = constants.%.10]
// CHECK:STDOUT:   %f_ref.var: ref %.10 = var f_ref
// CHECK:STDOUT:   %f_ref: ref %.10 = bind_name f_ref, %f_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %Self: %Basic.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc8: %.1 = assoc_entity element0, %T [template = constants.%.2]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc9: %.4 = assoc_entity element1, %F.decl [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc8
// CHECK:STDOUT:   .F = %.loc9
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc16: %.6 = assoc_entity element0, %T [template = constants.%.7]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc17: %.8 = assoc_entity element1, %F.decl [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc16
// CHECK:STDOUT:   .F = %.loc17
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Basic.%Self: %Basic.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self.1: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [template]
// CHECK:STDOUT:   %Basic.type: type = interface_type @Basic [template]
// CHECK:STDOUT:   %Self.2: %Basic.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [template]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [template]
// CHECK:STDOUT:   %ForwardDeclared.type: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %Self.3: %ForwardDeclared.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [template]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Basic.type, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, imports.%import_ref.16 [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %Basic.type, %F.type.1 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, imports.%import_ref.17 [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ForwardDeclared.type, type [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.18 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ForwardDeclared.type, %F.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element1, imports.%import_ref.19 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %ForwardDeclared.type [template]
// CHECK:STDOUT:   %.11: type = struct_type {.f: %ForwardDeclared.type} [template]
// CHECK:STDOUT:   %.12: type = struct_type {.f: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+1, loaded [template = constants.%Empty.type]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+5, loaded [template = constants.%Basic.type]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//a, inst+20, loaded [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %import_ref.4: ref %.11 = import_ref Main//a, inst+42, loaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//a, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.7: %.2 = import_ref Main//a, inst+11, loaded [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.8: %.4 = import_ref Main//a, inst+18, loaded [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//a, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//a, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//a, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.12: %.6 = import_ref Main//a, inst+26, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.13: %.8 = import_ref Main//a, inst+32, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//a, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//a, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//a, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//a, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//a, inst+24, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//a, inst+28, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = imports.%import_ref.1
// CHECK:STDOUT:     .Basic = imports.%import_ref.2
// CHECK:STDOUT:     .ForwardDeclared = imports.%import_ref.3
// CHECK:STDOUT:     .f_ref = imports.%import_ref.4
// CHECK:STDOUT:     .UseEmpty = %UseEmpty.decl
// CHECK:STDOUT:     .UseBasic = %UseBasic.decl
// CHECK:STDOUT:     .UseForwardDeclared = %UseForwardDeclared.decl
// CHECK:STDOUT:     .UseBasicT = %UseBasicT
// CHECK:STDOUT:     .UseBasicF = %UseBasicF
// CHECK:STDOUT:     .UseForwardDeclaredT = %UseForwardDeclaredT
// CHECK:STDOUT:     .UseForwardDeclaredF = %UseForwardDeclaredF
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UseEmpty.decl: %UseEmpty.type = fn_decl @UseEmpty [template = constants.%UseEmpty] {
// CHECK:STDOUT:     %e.patt: %Empty.type = binding_pattern e
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%import_ref.1 [template = constants.%Empty.type]
// CHECK:STDOUT:     %e.param: %Empty.type = param e, runtime_param0
// CHECK:STDOUT:     %e: %Empty.type = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [template = constants.%UseBasic] {
// CHECK:STDOUT:     %e.patt: %Basic.type = binding_pattern e
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Basic.ref: type = name_ref Basic, imports.%import_ref.2 [template = constants.%Basic.type]
// CHECK:STDOUT:     %e.param: %Basic.type = param e, runtime_param0
// CHECK:STDOUT:     %e: %Basic.type = bind_name e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [template = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %f.patt: %ForwardDeclared.type = binding_pattern f
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ForwardDeclared.ref: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:     %f.param: %ForwardDeclared.type = param f, runtime_param0
// CHECK:STDOUT:     %f: %ForwardDeclared.type = bind_name f, %f.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%import_ref.2 [template = constants.%Basic.type]
// CHECK:STDOUT:   %T.ref.loc10: %.2 = name_ref T, imports.%import_ref.7 [template = constants.%.3]
// CHECK:STDOUT:   %UseBasicT: %.2 = bind_alias UseBasicT, imports.%import_ref.7 [template = constants.%.3]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%import_ref.2 [template = constants.%Basic.type]
// CHECK:STDOUT:   %F.ref.loc11: %.4 = name_ref F, imports.%import_ref.8 [template = constants.%.5]
// CHECK:STDOUT:   %UseBasicF: %.4 = bind_alias UseBasicF, imports.%import_ref.8 [template = constants.%.5]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %T.ref.loc13: %.6 = name_ref T, imports.%import_ref.12 [template = constants.%.7]
// CHECK:STDOUT:   %UseForwardDeclaredT: %.6 = bind_alias UseForwardDeclaredT, imports.%import_ref.12 [template = constants.%.7]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %F.ref.loc14: %.8 = name_ref F, imports.%import_ref.13 [template = constants.%.9]
// CHECK:STDOUT:   %UseForwardDeclaredF: %.8 = bind_alias UseForwardDeclaredF, imports.%import_ref.13 [template = constants.%.9]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%ForwardDeclared.type]
// CHECK:STDOUT:   %.loc16: type = ptr_type %ForwardDeclared.type [template = constants.%.10]
// CHECK:STDOUT:   %f.var: ref %.10 = var f
// CHECK:STDOUT:   %f: ref %.10 = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   .T = imports.%import_ref.7
// CHECK:STDOUT:   .F = imports.%import_ref.8
// CHECK:STDOUT:   witness = (imports.%import_ref.9, imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.11
// CHECK:STDOUT:   .T = imports.%import_ref.12
// CHECK:STDOUT:   .F = imports.%import_ref.13
// CHECK:STDOUT:   witness = (imports.%import_ref.14, imports.%import_ref.15)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e: %Empty.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e: %Basic.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f: %ForwardDeclared.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self.2: %Basic.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(constants.%Self.3: %ForwardDeclared.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref %.11 = name_ref f_ref, imports.%import_ref.4
// CHECK:STDOUT:   %.loc16_33: ref %ForwardDeclared.type = struct_access %f_ref.ref, element0
// CHECK:STDOUT:   %.loc16_27: %.10 = addr_of %.loc16_33
// CHECK:STDOUT:   assign file.%f.var, %.loc16_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.3) {}
// CHECK:STDOUT:

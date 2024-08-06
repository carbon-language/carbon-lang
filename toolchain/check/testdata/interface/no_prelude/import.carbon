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

library "a";

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

library "b";

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
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @Basic [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Basic.%T [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @Basic[%Self.2] [symbolic]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element1, @Basic.%F.decl [symbolic]
// CHECK:STDOUT:   %.8: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %Self.3: %.8 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, @ForwardDeclared.%T [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @ForwardDeclared[%Self.3] [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.8, %F.type.2 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element1, @ForwardDeclared.%F.decl [symbolic]
// CHECK:STDOUT:   %.13: type = struct_type {.f: %.8} [template]
// CHECK:STDOUT:   %.14: type = struct_type {.f: %.5} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Basic = %Basic.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl
// CHECK:STDOUT:     .f_ref = %f_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:   %Basic.decl: type = interface_decl @Basic [template = constants.%.2] {}
// CHECK:STDOUT:   %ForwardDeclared.decl: type = interface_decl @ForwardDeclared [template = constants.%.8] {}
// CHECK:STDOUT:   %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl [template = constants.%.8]
// CHECK:STDOUT:   %.loc20: type = struct_type {.f: %.8} [template = constants.%.13]
// CHECK:STDOUT:   %f_ref.var: ref %.13 = var f_ref
// CHECK:STDOUT:   %f_ref: ref %.13 = bind_name f_ref, %f_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic[%Self.1: %.2] {
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Basic[%Self.2] [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @Basic.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.2, @Basic.%F.type (%F.type.1) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %.2: @Basic.%.1 (%.6) = assoc_entity element1, %F.decl [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc8: %.3 = assoc_entity element0, %T [template = constants.%.4]
// CHECK:STDOUT:   %F.decl: @Basic.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = %F (constants.%F.1)] {}
// CHECK:STDOUT:   %.loc9: @Basic.%.1 (%.6) = assoc_entity element1, %F.decl [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .T = %.loc8
// CHECK:STDOUT:   .F = %.loc9
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared[%Self.1: %.8] {
// CHECK:STDOUT:   %Self.2: %.8 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @ForwardDeclared[%Self.2] [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @ForwardDeclared.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.8, @ForwardDeclared.%F.type (%F.type.2) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @ForwardDeclared.%.1 (%.11) = assoc_entity element1, %F.decl [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.8 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc16: %.9 = assoc_entity element0, %T [template = constants.%.10]
// CHECK:STDOUT:   %F.decl: @ForwardDeclared.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = %F (constants.%F.2)] {}
// CHECK:STDOUT:   %.loc17: @ForwardDeclared.%.1 (%.11) = assoc_entity element1, %F.decl [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .T = %.loc16
// CHECK:STDOUT:   .F = %.loc17
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Basic.%Self.1: %.2) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@ForwardDeclared.%Self.1: %.8) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty[constants.%Self.1] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic[constants.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic[@Basic.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[constants.%Self.3] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%Self.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[@ForwardDeclared.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %UseEmpty.type: type = fn_type @UseEmpty [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %UseEmpty: %UseEmpty.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = interface_type @Basic [template]
// CHECK:STDOUT:   %Self.2: %.3 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @Basic[%Self.2] [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.3, %F.type.2 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element1, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %UseBasic.type: type = fn_type @UseBasic [template]
// CHECK:STDOUT:   %UseBasic: %UseBasic.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %Self.3: %.6 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.2, @ForwardDeclared[%Self.3] [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.6, %F.type.4 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element1, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %UseForwardDeclared.type: type = fn_type @UseForwardDeclared [template]
// CHECK:STDOUT:   %UseForwardDeclared: %UseForwardDeclared.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.3, type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.18 [template]
// CHECK:STDOUT:   %.11: %.4 = assoc_entity element1, imports.%import_ref.19 [symbolic]
// CHECK:STDOUT:   %.12: type = assoc_entity_type %.6, type [template]
// CHECK:STDOUT:   %.13: %.12 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %.14: %.7 = assoc_entity element1, imports.%import_ref.21 [symbolic]
// CHECK:STDOUT:   %.15: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %.16: type = struct_type {.f: %.6} [template]
// CHECK:STDOUT:   %.17: type = struct_type {.f: %.2} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+1, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+6, loaded [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3: type = import_ref Main//a, inst+26, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4: ref %.16 = import_ref Main//a, inst+53, loaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//a, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//a, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//a, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.8: %.9 = import_ref Main//a, inst+13, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.9: @Basic.%.1 (%.4) = import_ref Main//a, inst+20, loaded [symbolic = @Basic.%.2 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//a, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//a, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//a, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//a, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.14: %.12 = import_ref Main//a, inst+33, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.15: @ForwardDeclared.%.1 (%.7) = import_ref Main//a, inst+39, loaded [symbolic = @ForwardDeclared.%.2 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//a, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//a, inst+35, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//a, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//a, inst+15, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//a, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//a, inst+35, unloaded
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
// CHECK:STDOUT:     .f = %f.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %UseEmpty.decl: %UseEmpty.type = fn_decl @UseEmpty [template = constants.%UseEmpty] {
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, imports.%import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:     %e.loc6_13.1: %.1 = param e
// CHECK:STDOUT:     @UseEmpty.%e: %.1 = bind_name e, %e.loc6_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic.decl: %UseBasic.type = fn_decl @UseBasic [template = constants.%UseBasic] {
// CHECK:STDOUT:     %Basic.ref.loc7: type = name_ref Basic, imports.%import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:     %e.loc7_13.1: %.3 = param e
// CHECK:STDOUT:     @UseBasic.%e: %.3 = bind_name e, %e.loc7_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared.decl: %UseForwardDeclared.type = fn_decl @UseForwardDeclared [template = constants.%UseForwardDeclared] {
// CHECK:STDOUT:     %ForwardDeclared.ref.loc8: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%.6]
// CHECK:STDOUT:     %f.loc8_23.1: %.6 = param f
// CHECK:STDOUT:     @UseForwardDeclared.%f: %.6 = bind_name f, %f.loc8_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, imports.%import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:   %T.ref.loc10: %.9 = name_ref T, imports.%import_ref.8 [template = constants.%.10]
// CHECK:STDOUT:   %UseBasicT: %.9 = bind_alias UseBasicT, imports.%import_ref.8 [template = constants.%.10]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, imports.%import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:   %F.ref.loc11: %.4 = name_ref F, imports.%import_ref.9 [symbolic = constants.%.11]
// CHECK:STDOUT:   %UseBasicF: %.4 = bind_alias UseBasicF, imports.%import_ref.9 [symbolic = constants.%.11]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%.6]
// CHECK:STDOUT:   %T.ref.loc13: %.12 = name_ref T, imports.%import_ref.14 [template = constants.%.13]
// CHECK:STDOUT:   %UseForwardDeclaredT: %.12 = bind_alias UseForwardDeclaredT, imports.%import_ref.14 [template = constants.%.13]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%.6]
// CHECK:STDOUT:   %F.ref.loc14: %.7 = name_ref F, imports.%import_ref.15 [symbolic = constants.%.14]
// CHECK:STDOUT:   %UseForwardDeclaredF: %.7 = bind_alias UseForwardDeclaredF, imports.%import_ref.15 [symbolic = constants.%.14]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, imports.%import_ref.3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16: type = ptr_type %.6 [template = constants.%.15]
// CHECK:STDOUT:   %f.var: ref %.15 = var f
// CHECK:STDOUT:   %f.loc16: ref %.15 = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty[constants.%Self.1: %.1] {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.5
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic[constants.%Self.2: %.3] {
// CHECK:STDOUT:   %Self: %.3 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Basic[%Self] [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @Basic.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.3, @Basic.%F.type (%F.type.2) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @Basic.%.1 (%.4) = assoc_entity element1, imports.%import_ref.6 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT:   .T = imports.%import_ref.8
// CHECK:STDOUT:   .F = imports.%import_ref.9
// CHECK:STDOUT:   witness = (imports.%import_ref.10, imports.%import_ref.11)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared[constants.%Self.3: %.6] {
// CHECK:STDOUT:   %Self: %.6 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @ForwardDeclared[%Self] [symbolic = %F.type (constants.%F.type.4)]
// CHECK:STDOUT:   %F: @ForwardDeclared.%F.type (%F.type.4) = struct_value () [symbolic = %F (constants.%F.4)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.6, @ForwardDeclared.%F.type (%F.type.4) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %.2: @ForwardDeclared.%.1 (%.7) = assoc_entity element1, imports.%import_ref.12 [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.13
// CHECK:STDOUT:   .T = imports.%import_ref.14
// CHECK:STDOUT:   .F = imports.%import_ref.15
// CHECK:STDOUT:   witness = (imports.%import_ref.16, imports.%import_ref.17)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%Self.2: %.3) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(constants.%Self.3: %.6) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f: %.6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref %.16 = name_ref f_ref, imports.%import_ref.4
// CHECK:STDOUT:   %.loc16_33: ref %.6 = struct_access %f_ref.ref, element0
// CHECK:STDOUT:   %.loc16_27: %.15 = addr_of %.loc16_33
// CHECK:STDOUT:   assign file.%f.var, %.loc16_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic[constants.%Self.2] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Basic[@Basic.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[constants.%Self.3] {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[@ForwardDeclared.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:

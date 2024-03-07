// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntityType`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntity`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntityType`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntity`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntityType`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntity`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntityType`.
// CHECK:STDERR: b.carbon: ERROR: Semantics TODO: `TryResolveInst on AssociatedEntity`.

// --- a.carbon

library "a" api;

interface Empty {
}

interface Basic {
  let T:! type;
  fn F();
}

interface ForwardDeclared;

interface ForwardDeclared {
  let T:! type;
  fn F();
}

var f_ref: {.f: ForwardDeclared};

// --- b.carbon

library "b" api;

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
// CHECK:STDOUT:   %.2: type = interface_type @Basic [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Basic, type [template]
// CHECK:STDOUT:   %.4: <associated type in Basic> = assoc_entity element0, @Basic.%T [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Basic, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Basic> = assoc_entity element1, @Basic.%F [template]
// CHECK:STDOUT:   %.7: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @ForwardDeclared, type [template]
// CHECK:STDOUT:   %.9: <associated type in ForwardDeclared> = assoc_entity element0, @ForwardDeclared.%T [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type @ForwardDeclared, <function> [template]
// CHECK:STDOUT:   %.11: <associated <function> in ForwardDeclared> = assoc_entity element1, @ForwardDeclared.%F [template]
// CHECK:STDOUT:   %.12: type = struct_type {.f: ForwardDeclared} [template]
// CHECK:STDOUT:   %.13: type = tuple_type () [template]
// CHECK:STDOUT:   %.14: type = struct_type {.f: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Basic = %Basic.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc12
// CHECK:STDOUT:     .f_ref = %f_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:   %Basic.decl: type = interface_decl @Basic [template = constants.%.2] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc12: type = interface_decl @ForwardDeclared [template = constants.%.7] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc14: type = interface_decl @ForwardDeclared [template = constants.%.7] {}
// CHECK:STDOUT:   %ForwardDeclared.ref: type = name_ref ForwardDeclared, %ForwardDeclared.decl.loc12 [template = constants.%.7]
// CHECK:STDOUT:   %.loc19: type = struct_type {.f: ForwardDeclared} [template = constants.%.12]
// CHECK:STDOUT:   %f_ref.var: ref {.f: ForwardDeclared} = var f_ref
// CHECK:STDOUT:   %f_ref: ref {.f: ForwardDeclared} = bind_name f_ref, %f_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: Empty = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %Self: Basic = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc8: <associated type in Basic> = assoc_entity element0, %T [template = constants.%.4]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %.loc9: <associated <function> in Basic> = assoc_entity element1, %F [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc8
// CHECK:STDOUT:   .F = %.loc9
// CHECK:STDOUT:   witness = (%T, %F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: ForwardDeclared = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc15: <associated type in ForwardDeclared> = assoc_entity element0, %T [template = constants.%.9]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {}
// CHECK:STDOUT:   %.loc16: <associated <function> in ForwardDeclared> = assoc_entity element1, %F [template = constants.%.11]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc15
// CHECK:STDOUT:   .F = %.loc16
// CHECK:STDOUT:   witness = (%T, %F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = interface_type @Basic [template]
// CHECK:STDOUT:   %.4: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %.5: type = ptr_type ForwardDeclared [template]
// CHECK:STDOUT:   %.6: type = struct_type {.f: ForwardDeclared} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.f: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %import_ref.1
// CHECK:STDOUT:     .Basic = %import_ref.2
// CHECK:STDOUT:     .ForwardDeclared = %import_ref.3
// CHECK:STDOUT:     .f_ref = %import_ref.4
// CHECK:STDOUT:     .UseEmpty = %UseEmpty
// CHECK:STDOUT:     .UseBasic = %UseBasic
// CHECK:STDOUT:     .UseForwardDeclared = %UseForwardDeclared
// CHECK:STDOUT:     .UseBasicT = %UseBasicT
// CHECK:STDOUT:     .UseBasicF = %UseBasicF
// CHECK:STDOUT:     .UseForwardDeclaredT = %UseForwardDeclaredT
// CHECK:STDOUT:     .UseForwardDeclaredF = %UseForwardDeclaredF
// CHECK:STDOUT:     .f = %f.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+4, used [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+15, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.4: ref {.f: ForwardDeclared} = import_ref ir1, inst+36, used
// CHECK:STDOUT:   %UseEmpty: <function> = fn_decl @UseEmpty [template] {
// CHECK:STDOUT:     %Empty.decl: invalid = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, %import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:     %e.loc6_13.1: Empty = param e
// CHECK:STDOUT:     @UseEmpty.%e: Empty = bind_name e, %e.loc6_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseBasic: <function> = fn_decl @UseBasic [template] {
// CHECK:STDOUT:     %Basic.decl: invalid = interface_decl @Basic [template = constants.%.3] {}
// CHECK:STDOUT:     %Basic.ref.loc7: type = name_ref Basic, %import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:     %e.loc7_13.1: Basic = param e
// CHECK:STDOUT:     @UseBasic.%e: Basic = bind_name e, %e.loc7_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseForwardDeclared: <function> = fn_decl @UseForwardDeclared [template] {
// CHECK:STDOUT:     %ForwardDeclared.decl: invalid = interface_decl @ForwardDeclared [template = constants.%.4] {}
// CHECK:STDOUT:     %ForwardDeclared.ref.loc8: type = name_ref ForwardDeclared, %import_ref.3 [template = constants.%.4]
// CHECK:STDOUT:     %f.loc8_23.1: ForwardDeclared = param f
// CHECK:STDOUT:     @UseForwardDeclared.%f: ForwardDeclared = bind_name f, %f.loc8_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Basic.ref.loc10: type = name_ref Basic, %import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:   %T.ref.loc10: <error> = name_ref T, @Basic.%import_ref.3 [template = <error>]
// CHECK:STDOUT:   %UseBasicT: <error> = bind_alias UseBasicT, @Basic.%import_ref.3 [template = <error>]
// CHECK:STDOUT:   %Basic.ref.loc11: type = name_ref Basic, %import_ref.2 [template = constants.%.3]
// CHECK:STDOUT:   %F.ref.loc11: <error> = name_ref F, @Basic.%import_ref.2 [template = <error>]
// CHECK:STDOUT:   %UseBasicF: <error> = bind_alias UseBasicF, @Basic.%import_ref.2 [template = <error>]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc13: type = name_ref ForwardDeclared, %import_ref.3 [template = constants.%.4]
// CHECK:STDOUT:   %T.ref.loc13: <error> = name_ref T, @ForwardDeclared.%import_ref.3 [template = <error>]
// CHECK:STDOUT:   %UseForwardDeclaredT: <error> = bind_alias UseForwardDeclaredT, @ForwardDeclared.%import_ref.3 [template = <error>]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc14: type = name_ref ForwardDeclared, %import_ref.3 [template = constants.%.4]
// CHECK:STDOUT:   %F.ref.loc14: <error> = name_ref F, @ForwardDeclared.%import_ref.2 [template = <error>]
// CHECK:STDOUT:   %UseForwardDeclaredF: <error> = bind_alias UseForwardDeclaredF, @ForwardDeclared.%import_ref.2 [template = <error>]
// CHECK:STDOUT:   %ForwardDeclared.ref.loc16: type = name_ref ForwardDeclared, %import_ref.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16: type = ptr_type ForwardDeclared [template = constants.%.5]
// CHECK:STDOUT:   %f.var: ref ForwardDeclared* = var f
// CHECK:STDOUT:   %f.loc16: ref ForwardDeclared* = bind_name f, %f.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+3, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %import_ref
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Basic {
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+6, unused
// CHECK:STDOUT:   %import_ref.2: <error> = import_ref ir1, inst+13, used [template = <error>]
// CHECK:STDOUT:   %import_ref.3: <error> = import_ref ir1, inst+9, used [template = <error>]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unused
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+11, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %import_ref.1
// CHECK:STDOUT:   .F = %import_ref.2
// CHECK:STDOUT:   .T = %import_ref.3
// CHECK:STDOUT:   witness = (%import_ref.4, %import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   %import_ref.2: <error> = import_ref ir1, inst+25, used [template = <error>]
// CHECK:STDOUT:   %import_ref.3: <error> = import_ref ir1, inst+21, used [template = <error>]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unused
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+23, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %import_ref.1
// CHECK:STDOUT:   .F = %import_ref.2
// CHECK:STDOUT:   .T = %import_ref.3
// CHECK:STDOUT:   witness = (%import_ref.4, %import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseEmpty(%e: Empty) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseBasic(%e: Basic) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseForwardDeclared(%f: ForwardDeclared) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %f_ref.ref: ref {.f: ForwardDeclared} = name_ref f_ref, file.%import_ref.4
// CHECK:STDOUT:   %.loc16_33: ref ForwardDeclared = struct_access %f_ref.ref, element0
// CHECK:STDOUT:   %.loc16_27: ForwardDeclared* = addr_of %.loc16_33
// CHECK:STDOUT:   assign file.%f.var, %.loc16_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

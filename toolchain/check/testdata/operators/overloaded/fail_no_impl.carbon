// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package User;

class C {};

fn TestUnary(a: C) -> C {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Negate in type C that does not implement that interface.
  // CHECK:STDERR:   return -a;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR:
  return -a;
}

fn TestBinary(a: C, b: C) -> C {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Add in type C that does not implement that interface.
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestRef(b: C) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:3: ERROR: Cannot access member of interface AddAssign in type C that does not implement that interface.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a += b;
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:3: ERROR: Cannot access member of interface Inc in type C that does not implement that interface.
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:   ^~~
  ++a;
}

// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %TestUnary: type = fn_type @TestUnary [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: TestUnary = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self.1: Negate = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %struct.2: Op = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Negate, Op [template]
// CHECK:STDOUT:   %.6: <associated Op in Negate> = assoc_entity element0, file.%import_ref.5 [template]
// CHECK:STDOUT:   %TestBinary: type = fn_type @TestBinary [template]
// CHECK:STDOUT:   %struct.3: TestBinary = struct_value () [template]
// CHECK:STDOUT:   %.7: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self.2: Add = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %struct.4: Op = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Add, Op [template]
// CHECK:STDOUT:   %.9: <associated Op in Add> = assoc_entity element0, file.%import_ref.10 [template]
// CHECK:STDOUT:   %TestRef: type = fn_type @TestRef [template]
// CHECK:STDOUT:   %struct.5: TestRef = struct_value () [template]
// CHECK:STDOUT:   %struct.6: C = struct_value () [template]
// CHECK:STDOUT:   %.10: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.3: AddAssign = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.11: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %Op.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %struct.7: Op = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type @AddAssign, Op [template]
// CHECK:STDOUT:   %.13: <associated Op in AddAssign> = assoc_entity element0, file.%import_ref.15 [template]
// CHECK:STDOUT:   %.14: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self.4: Inc = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.15: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %Op.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %struct.8: Op = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type @Inc, Op [template]
// CHECK:STDOUT:   %.17: <associated Op in Inc> = assoc_entity element0, file.%import_ref.20 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestUnary = %TestUnary.decl
// CHECK:STDOUT:     .TestBinary = %TestBinary.decl
// CHECK:STDOUT:     .TestRef = %TestRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %TestUnary.decl: TestUnary = fn_decl @TestUnary [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc11_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc11_14.1: C = param a
// CHECK:STDOUT:     @TestUnary.%a: C = bind_name a, %a.loc11_14.1
// CHECK:STDOUT:     %C.ref.loc11_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestUnary.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir4, inst+67, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir4, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Op in Negate> = import_ref ir4, inst+84, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir4, inst+80, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir4, inst+80, unloaded
// CHECK:STDOUT:   %TestBinary.decl: TestBinary = fn_decl @TestBinary [template = constants.%struct.3] {
// CHECK:STDOUT:     %C.ref.loc19_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc19_15.1: C = param a
// CHECK:STDOUT:     @TestBinary.%a: C = bind_name a, %a.loc19_15.1
// CHECK:STDOUT:     %C.ref.loc19_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc19_21.1: C = param b
// CHECK:STDOUT:     @TestBinary.%b: C = bind_name b, %b.loc19_21.1
// CHECK:STDOUT:     %C.ref.loc19_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestBinary.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: type = import_ref ir4, inst+1, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir4, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.8: <associated Op in Add> = import_ref ir4, inst+24, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.9 = import_ref ir4, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir4, inst+19, unloaded
// CHECK:STDOUT:   %TestRef.decl: TestRef = fn_decl @TestRef [template = constants.%struct.5] {
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc27_12.1: C = param b
// CHECK:STDOUT:     @TestRef.%b: C = bind_name b, %b.loc27_12.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir4, inst+26, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir4, inst+28, unloaded
// CHECK:STDOUT:   %import_ref.13: <associated Op in AddAssign> = import_ref ir4, inst+47, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.14 = import_ref ir4, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref ir4, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref ir4, inst+49, loaded [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.17 = import_ref ir4, inst+51, unloaded
// CHECK:STDOUT:   %import_ref.18: <associated Op in Inc> = import_ref ir4, inst+65, loaded [template = constants.%.17]
// CHECK:STDOUT:   %import_ref.19 = import_ref ir4, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref ir4, inst+61, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Op = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT:   .Op = file.%import_ref.8
// CHECK:STDOUT:   witness = (file.%import_ref.9)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.12
// CHECK:STDOUT:   .Op = file.%import_ref.13
// CHECK:STDOUT:   witness = (file.%import_ref.14)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.17
// CHECK:STDOUT:   .Op = file.%import_ref.18
// CHECK:STDOUT:   witness = (file.%import_ref.19)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnary(%a: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a: C, %b: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref C = var a
// CHECK:STDOUT:   %a: ref C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc28_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc28_15.2: init C = class_init (), %a.var [template = constants.%struct.6]
// CHECK:STDOUT:   %.loc28_16: init C = converted %.loc28_15.1, %.loc28_15.2 [template = constants.%struct.6]
// CHECK:STDOUT:   assign %a.var, %.loc28_16
// CHECK:STDOUT:   %a.ref.loc33: ref C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc37: ref C = name_ref a, %a
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr %self: Self*](%other: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self: Self*]();
// CHECK:STDOUT:

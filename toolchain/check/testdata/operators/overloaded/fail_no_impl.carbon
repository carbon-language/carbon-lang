// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Negate {
  fn Op[self: Self]();
}
interface Add {
  fn Op[self: Self](other: Self) -> Self;
}
interface AddAssign {
  fn Op[addr self: Self*](other: Self);
}
interface Inc {
  fn Op[addr self: Self*]();
}

// --- fail_no_impl.carbon

package User api;

import Core;

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

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Negate [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Negate, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Negate> = assoc_entity element0, @Negate.%Op [template]
// CHECK:STDOUT:   %.4: type = interface_type @Add [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Add, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Add> = assoc_entity element0, @Add.%Op [template]
// CHECK:STDOUT:   %.7: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %.8: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type @AddAssign, <function> [template]
// CHECK:STDOUT:   %.10: <associated <function> in AddAssign> = assoc_entity element0, @AddAssign.%Op [template]
// CHECK:STDOUT:   %.11: type = interface_type @Inc [template]
// CHECK:STDOUT:   %.12: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.13: type = assoc_entity_type @Inc, <function> [template]
// CHECK:STDOUT:   %.14: <associated <function> in Inc> = assoc_entity element0, @Inc.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .AddAssign = %AddAssign.decl
// CHECK:STDOUT:     .Inc = %Inc.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: type = interface_decl @Negate [template = constants.%.1] {}
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%.4] {}
// CHECK:STDOUT:   %AddAssign.decl: type = interface_decl @AddAssign [template = constants.%.7] {}
// CHECK:STDOUT:   %Inc.decl: type = interface_decl @Inc [template = constants.%.11] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT:   %Self: Negate = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %Self.ref: Negate = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref, %.loc5_15.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_22: <associated <function> in Negate> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_22
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: Add = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_15: Add = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_15.1: type = facet_type_access %Self.ref.loc8_15 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_15.2: type = converted %Self.ref.loc8_15, %.loc8_15.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc8_9.1: Self = param self
// CHECK:STDOUT:     %self.loc8_9.2: Self = bind_name self, %self.loc8_9.1
// CHECK:STDOUT:     %Self.ref.loc8_28: Add = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_28.1: type = facet_type_access %Self.ref.loc8_28 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %Self.ref.loc8_28, %.loc8_28.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_21.1: Self = param other
// CHECK:STDOUT:     %other.loc8_21.2: Self = bind_name other, %other.loc8_21.1
// CHECK:STDOUT:     %Self.ref.loc8_37: Add = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_37.1: type = facet_type_access %Self.ref.loc8_37 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %Self.ref.loc8_37, %.loc8_37.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_41: <associated <function> in Add> = assoc_entity element0, %Op [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8_41
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT:   %Self: AddAssign = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.3 [template] {
// CHECK:STDOUT:     %Self.ref.loc11_20: AddAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc11_24.1: type = facet_type_access %Self.ref.loc11_20 [symbolic = %Self]
// CHECK:STDOUT:     %.loc11_20: type = converted %Self.ref.loc11_20, %.loc11_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc11_24.2: type = ptr_type Self [symbolic = constants.%.8]
// CHECK:STDOUT:     %self.loc11_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc11_14.3: Self* = bind_name self, %self.loc11_14.1
// CHECK:STDOUT:     %.loc11_9: Self* = addr_pattern %self.loc11_14.3
// CHECK:STDOUT:     %Self.ref.loc11_34: AddAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc11_34.1: type = facet_type_access %Self.ref.loc11_34 [symbolic = %Self]
// CHECK:STDOUT:     %.loc11_34.2: type = converted %Self.ref.loc11_34, %.loc11_34.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc11_27.1: Self = param other
// CHECK:STDOUT:     %other.loc11_27.2: Self = bind_name other, %other.loc11_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_39: <associated <function> in AddAssign> = assoc_entity element0, %Op [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc11_39
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT:   %Self: Inc = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.4 [template] {
// CHECK:STDOUT:     %Self.ref: Inc = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc14_24.1: type = facet_type_access %Self.ref [symbolic = %Self]
// CHECK:STDOUT:     %.loc14_20: type = converted %Self.ref, %.loc14_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc14_24.2: type = ptr_type Self [symbolic = constants.%.12]
// CHECK:STDOUT:     %self.loc14_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc14_14.3: Self* = bind_name self, %self.loc14_14.1
// CHECK:STDOUT:     %.loc14_9: Self* = addr_pattern %self.loc14_14.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_28: <associated <function> in Inc> = assoc_entity element0, %Op [template = constants.%.14]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc14_28
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@Negate.%self.loc5_9.2: Self]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[@Add.%self.loc8_9.2: Self](@Add.%other.loc8_21.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr @AddAssign.%self.loc11_14.3: Self*](@AddAssign.%other.loc11_27.2: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr @Inc.%self.loc14_14.3: Self*]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Negate [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Negate, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Negate> = assoc_entity element0, file.%import_ref.5 [template]
// CHECK:STDOUT:   %.7: type = interface_type @Add [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Add, <function> [template]
// CHECK:STDOUT:   %.9: <associated <function> in Add> = assoc_entity element0, file.%import_ref.10 [template]
// CHECK:STDOUT:   %.10: C = struct_value () [template]
// CHECK:STDOUT:   %.11: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type @AddAssign, <function> [template]
// CHECK:STDOUT:   %.13: <associated <function> in AddAssign> = assoc_entity element0, file.%import_ref.15 [template]
// CHECK:STDOUT:   %.14: type = interface_type @Inc [template]
// CHECK:STDOUT:   %.15: type = assoc_entity_type @Inc, <function> [template]
// CHECK:STDOUT:   %.16: <associated <function> in Inc> = assoc_entity element0, file.%import_ref.20 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestUnary = %TestUnary
// CHECK:STDOUT:     .TestBinary = %TestBinary
// CHECK:STDOUT:     .TestRef = %TestRef
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %TestUnary: <function> = fn_decl @TestUnary [template] {
// CHECK:STDOUT:     %C.ref.loc8_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc8_14.1: C = param a
// CHECK:STDOUT:     @TestUnary.%a: C = bind_name a, %a.loc8_14.1
// CHECK:STDOUT:     %C.ref.loc8_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestUnary.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Negate> = import_ref ir1, inst+11, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+9, unused
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+9, unused
// CHECK:STDOUT:   %TestBinary: <function> = fn_decl @TestBinary [template] {
// CHECK:STDOUT:     %C.ref.loc16_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc16_15.1: C = param a
// CHECK:STDOUT:     @TestBinary.%a: C = bind_name a, %a.loc16_15.1
// CHECK:STDOUT:     %C.ref.loc16_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc16_21.1: C = param b
// CHECK:STDOUT:     @TestBinary.%b: C = bind_name b, %b.loc16_21.1
// CHECK:STDOUT:     %C.ref.loc16_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestBinary.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: type = import_ref ir1, inst+13, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.7: <associated <function> in Add> = import_ref ir1, inst+32, used [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %TestRef: <function> = fn_decl @TestRef [template] {
// CHECK:STDOUT:     %C.ref.loc24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc24_12.1: C = param b
// CHECK:STDOUT:     @TestRef.%b: C = bind_name b, %b.loc24_12.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir1, inst+34, used [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.12: <associated <function> in AddAssign> = import_ref ir1, inst+52, used [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.13 = import_ref ir1, inst+36, unused
// CHECK:STDOUT:   %import_ref.14 = import_ref ir1, inst+50, unused
// CHECK:STDOUT:   %import_ref.15 = import_ref ir1, inst+50, unused
// CHECK:STDOUT:   %import_ref.16: type = import_ref ir1, inst+54, used [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.17: <associated <function> in Inc> = import_ref ir1, inst+67, used [template = constants.%.16]
// CHECK:STDOUT:   %import_ref.18 = import_ref ir1, inst+56, unused
// CHECK:STDOUT:   %import_ref.19 = import_ref ir1, inst+65, unused
// CHECK:STDOUT:   %import_ref.20 = import_ref ir1, inst+65, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.7
// CHECK:STDOUT:   .Self = file.%import_ref.8
// CHECK:STDOUT:   witness = (file.%import_ref.9)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.12
// CHECK:STDOUT:   .Self = file.%import_ref.13
// CHECK:STDOUT:   witness = (file.%import_ref.14)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.17
// CHECK:STDOUT:   .Self = file.%import_ref.18
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
// CHECK:STDOUT:   %Negate.decl: invalid = interface_decl @Negate [template = constants.%.4] {}
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a: C, %b: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %Add.decl: invalid = interface_decl @Add [template = constants.%.7] {}
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref C = var a
// CHECK:STDOUT:   %a: ref C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc25_15.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc25_15.2: init C = class_init (), %a.var [template = constants.%.10]
// CHECK:STDOUT:   %.loc25_15.3: init C = converted %.loc25_15.1, %.loc25_15.2 [template = constants.%.10]
// CHECK:STDOUT:   assign %a.var, %.loc25_15.3
// CHECK:STDOUT:   %a.ref.loc30: ref C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %AddAssign.decl: invalid = interface_decl @AddAssign [template = constants.%.11] {}
// CHECK:STDOUT:   %a.ref.loc34: ref C = name_ref a, %a
// CHECK:STDOUT:   %Inc.decl: invalid = interface_decl @Inc [template = constants.%.14] {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

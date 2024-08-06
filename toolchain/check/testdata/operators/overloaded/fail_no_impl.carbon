// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl.carbon

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
// CHECK:STDOUT:   %TestUnary.type: type = fn_type @TestUnary [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %TestUnary: %TestUnary.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.4: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self.1: %.4 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.1, @Negate[%Self.1] [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Op.type.2 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.2 [symbolic]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %TestBinary.type: type = fn_type @TestBinary [template]
// CHECK:STDOUT:   %TestBinary: %TestBinary.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self.2: %.8 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.2, @Add[%Self.2] [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, %Op.type.4 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.8 [symbolic]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: %.9 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %TestRef.type: type = fn_type @TestRef [template]
// CHECK:STDOUT:   %TestRef: %TestRef.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.12: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.3: %.12 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.5: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.5: %Op.type.5 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.6: type = fn_type @Op.3, @AddAssign[%Self.3] [symbolic]
// CHECK:STDOUT:   %.13: type = ptr_type %Self.3 [symbolic]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %.12, %Op.type.6 [symbolic]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.14 [symbolic]
// CHECK:STDOUT:   %Op.6: %Op.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: %.14 = assoc_entity element0, imports.%import_ref.18 [symbolic]
// CHECK:STDOUT:   %.17: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self.4: %.17 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.7: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.7: %Op.type.7 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.8: type = fn_type @Op.4, @Inc[%Self.4] [symbolic]
// CHECK:STDOUT:   %.18: type = ptr_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.19: type = assoc_entity_type %.17, %Op.type.8 [symbolic]
// CHECK:STDOUT:   %.20: %.19 = assoc_entity element0, imports.%import_ref.20 [symbolic]
// CHECK:STDOUT:   %Op.8: %Op.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.21: %.19 = assoc_entity element0, imports.%import_ref.24 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Negate = %import_ref.1
// CHECK:STDOUT:     .Add = %import_ref.7
// CHECK:STDOUT:     .AddAssign = %import_ref.13
// CHECK:STDOUT:     .Inc = %import_ref.19
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+93, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+109, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+95, unloaded
// CHECK:STDOUT:   %import_ref.4: @Negate.%.1 (%.5) = import_ref Core//prelude/operators/arithmetic, inst+114, loaded [symbolic = @Negate.%.2 (constants.%.7)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+109, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/arithmetic, inst+109, unloaded
// CHECK:STDOUT:   %import_ref.7: type = import_ref Core//prelude/operators/arithmetic, inst+1, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/arithmetic, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.10: @Add.%.1 (%.9) = import_ref Core//prelude/operators/arithmetic, inst+29, loaded [symbolic = @Add.%.2 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//prelude/operators/arithmetic, inst+35, loaded [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/arithmetic, inst+37, unloaded
// CHECK:STDOUT:   %import_ref.16: @AddAssign.%.1 (%.14) = import_ref Core//prelude/operators/arithmetic, inst+61, loaded [symbolic = @AddAssign.%.2 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/arithmetic, inst+67, loaded [template = constants.%.17]
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/arithmetic, inst+81, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/arithmetic, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.22: @Inc.%.1 (%.19) = import_ref Core//prelude/operators/arithmetic, inst+87, loaded [symbolic = @Inc.%.2 (constants.%.21)]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/arithmetic, inst+81, unloaded
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/arithmetic, inst+81, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestUnary = %TestUnary.decl
// CHECK:STDOUT:     .TestBinary = %TestBinary.decl
// CHECK:STDOUT:     .TestRef = %TestRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %TestUnary.decl: %TestUnary.type = fn_decl @TestUnary [template = constants.%TestUnary] {
// CHECK:STDOUT:     %C.ref.loc15_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_14.1: %C = param a
// CHECK:STDOUT:     @TestUnary.%a: %C = bind_name a, %a.loc15_14.1
// CHECK:STDOUT:     %C.ref.loc15_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestUnary.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBinary.decl: %TestBinary.type = fn_decl @TestBinary [template = constants.%TestBinary] {
// CHECK:STDOUT:     %C.ref.loc23_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc23_15.1: %C = param a
// CHECK:STDOUT:     @TestBinary.%a: %C = bind_name a, %a.loc23_15.1
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc23_21.1: %C = param b
// CHECK:STDOUT:     @TestBinary.%b: %C = bind_name b, %b.loc23_21.1
// CHECK:STDOUT:     %C.ref.loc23_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestBinary.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRef.decl: %TestRef.type = fn_decl @TestRef [template = constants.%TestRef] {
// CHECK:STDOUT:     %C.ref.loc31: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc31_12.1: %C = param b
// CHECK:STDOUT:     @TestRef.%b: %C = bind_name b, %b.loc31_12.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate[constants.%Self.1: %.4] {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Negate[%Self] [symbolic = %Op.type (constants.%Op.type.2)]
// CHECK:STDOUT:   %Op: @Negate.%Op.type (%Op.type.2) = struct_value () [symbolic = %Op (constants.%Op.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.4, @Negate.%Op.type (%Op.type.2) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @Negate.%.1 (%.5) = assoc_entity element0, imports.%import_ref.2 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add[constants.%Self.2: %.8] {
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @Add[%Self] [symbolic = %Op.type (constants.%Op.type.4)]
// CHECK:STDOUT:   %Op: @Add.%Op.type (%Op.type.4) = struct_value () [symbolic = %Op (constants.%Op.4)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.8, @Add.%Op.type (%Op.type.4) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @Add.%.1 (%.9) = assoc_entity element0, imports.%import_ref.8 [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.9
// CHECK:STDOUT:   .Op = imports.%import_ref.10
// CHECK:STDOUT:   witness = (imports.%import_ref.11)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign[constants.%Self.3: %.12] {
// CHECK:STDOUT:   %Self: %.12 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.3, @AddAssign[%Self] [symbolic = %Op.type (constants.%Op.type.6)]
// CHECK:STDOUT:   %Op: @AddAssign.%Op.type (%Op.type.6) = struct_value () [symbolic = %Op (constants.%Op.6)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.12, @AddAssign.%Op.type (%Op.type.6) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %.2: @AddAssign.%.1 (%.14) = assoc_entity element0, imports.%import_ref.14 [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.15
// CHECK:STDOUT:   .Op = imports.%import_ref.16
// CHECK:STDOUT:   witness = (imports.%import_ref.17)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc[constants.%Self.4: %.17] {
// CHECK:STDOUT:   %Self: %.17 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.4, @Inc[%Self] [symbolic = %Op.type (constants.%Op.type.8)]
// CHECK:STDOUT:   %Op: @Inc.%Op.type (%Op.type.8) = struct_value () [symbolic = %Op (constants.%Op.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.17, @Inc.%Op.type (%Op.type.8) [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %.2: @Inc.%.1 (%.19) = assoc_entity element0, imports.%import_ref.20 [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.21
// CHECK:STDOUT:   .Op = imports.%import_ref.22
// CHECK:STDOUT:   witness = (imports.%import_ref.23)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnary(%a: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.1: %.4) {
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self.1]() -> %Self.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a: %C, %b: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self.2: %.8) {
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self.2](%other: %Self.2) -> %Self.2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc32_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc32_15.2: init %C = class_init (), %a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc32_16: init %C = converted %.loc32_15.1, %.loc32_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc32_16
// CHECK:STDOUT:   %a.ref.loc37: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %a.ref.loc41: ref %C = name_ref a, %a
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.3(constants.%Self.3: %.12) {
// CHECK:STDOUT:   %Self: %.12 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.3.%Self (%Self.3) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: %.13](%other: %Self.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.4(constants.%Self.4: %.17) {
// CHECK:STDOUT:   %Self: %.17 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.4.%Self (%Self.4) [symbolic = %.2 (constants.%.18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: %.18]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Negate[constants.%Self.1] {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Negate[@Negate.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[constants.%Self.2] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[@Add.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[constants.%Self.3] {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[@AddAssign.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inc[constants.%Self.4] {
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inc[@Inc.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT: }
// CHECK:STDOUT:

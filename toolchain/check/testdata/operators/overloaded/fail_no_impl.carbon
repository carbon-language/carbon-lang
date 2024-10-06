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
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Negate` in type `C` that does not implement that interface
  // CHECK:STDERR:   return -a;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR:
  return -a;
}

fn TestBinary(a: C, b: C) -> C {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: error: cannot access member of interface `Add` in type `C` that does not implement that interface
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

fn TestRef(b: C) {
  var a: C = {};
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `AddAssign` in type `C` that does not implement that interface
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a += b;
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:3: error: cannot access member of interface `Inc` in type `C` that does not implement that interface
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:   ^~~
  ++a;
}

// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %TestUnary.type: type = fn_type @TestUnary [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %TestUnary: %TestUnary.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %Negate.type: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self.1: %Negate.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %Negate.type, %Op.type.1 [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %TestBinary.type: type = fn_type @TestBinary [template]
// CHECK:STDOUT:   %TestBinary: %TestBinary.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self.2: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %Add.type, %Op.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.10 [template]
// CHECK:STDOUT:   %TestRef.type: type = fn_type @TestRef [template]
// CHECK:STDOUT:   %TestRef: %TestRef.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %AddAssign.type: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.3: %AddAssign.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type %Self.3 [symbolic]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %AddAssign.type, %Op.type.3 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.15 [template]
// CHECK:STDOUT:   %Inc.type: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self.4: %Inc.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.12: type = ptr_type %Self.4 [symbolic]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %Inc.type, %Op.type.4 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Negate = %import_ref.1
// CHECK:STDOUT:     .Add = %import_ref.6
// CHECK:STDOUT:     .AddAssign = %import_ref.11
// CHECK:STDOUT:     .Inc = %import_ref.16
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+77, loaded [template = constants.%Negate.type]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+79, unloaded
// CHECK:STDOUT:   %import_ref.3: %.5 = import_ref Core//prelude/operators/arithmetic, inst+96, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/arithmetic, inst+91, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/arithmetic, inst+91, unloaded
// CHECK:STDOUT:   %import_ref.6: type = import_ref Core//prelude/operators/arithmetic, inst+1, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/arithmetic, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.8: %.7 = import_ref Core//prelude/operators/arithmetic, inst+27, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//prelude/operators/arithmetic, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/arithmetic, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/arithmetic, inst+29, loaded [template = constants.%AddAssign.type]
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/arithmetic, inst+31, unloaded
// CHECK:STDOUT:   %import_ref.13: %.10 = import_ref Core//prelude/operators/arithmetic, inst+54, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/arithmetic, inst+48, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Core//prelude/operators/arithmetic, inst+48, unloaded
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/arithmetic, inst+56, loaded [template = constants.%Inc.type]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/arithmetic, inst+58, unloaded
// CHECK:STDOUT:   %import_ref.18: %.13 = import_ref Core//prelude/operators/arithmetic, inst+75, loaded [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/arithmetic, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/arithmetic, inst+69, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %TestUnary.decl: %TestUnary.type = fn_decl @TestUnary [template = constants.%TestUnary] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %C.ref.loc15_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBinary.decl: %TestBinary.type = fn_decl @TestBinary [template = constants.%TestBinary] {
// CHECK:STDOUT:     %a.patt: %C = binding_pattern a
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc23_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.param: %C = param a, runtime_param0
// CHECK:STDOUT:     %a: %C = bind_name a, %a.param
// CHECK:STDOUT:     %C.ref.loc23_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.param: %C = param b, runtime_param1
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:     %C.ref.loc23_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRef.decl: %TestRef.type = fn_decl @TestRef [template = constants.%TestRef] {
// CHECK:STDOUT:     %b.patt: %C = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.param: %C = param b, runtime_param0
// CHECK:STDOUT:     %b: %C = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .Op = imports.%import_ref.3
// CHECK:STDOUT:   witness = (imports.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.7
// CHECK:STDOUT:   .Op = imports.%import_ref.8
// CHECK:STDOUT:   witness = (imports.%import_ref.9)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.12
// CHECK:STDOUT:   .Op = imports.%import_ref.13
// CHECK:STDOUT:   witness = (imports.%import_ref.14)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.17
// CHECK:STDOUT:   .Op = imports.%import_ref.18
// CHECK:STDOUT:   witness = (imports.%import_ref.19)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestUnary(%a: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %Op.ref: %.5 = name_ref Op, imports.%import_ref.3 [template = constants.%.6]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.1: %Negate.type) {
// CHECK:STDOUT:   %Self: %Negate.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.1.%Self (%Self.1)]() -> @Op.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBinary(%a: %C, %b: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Op.ref: %.7 = name_ref Op, imports.%import_ref.8 [template = constants.%.8]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(constants.%Self.2: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Op.2.%Self (%Self.2)](%other: @Op.2.%Self (%Self.2)) -> @Op.2.%Self (%Self.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRef(%b: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %a: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %.loc32_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc32_15.2: init %C = class_init (), %a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc32_16: init %C = converted %.loc32_15.1, %.loc32_15.2 [template = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc32_16
// CHECK:STDOUT:   %a.ref.loc37: ref %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %Op.ref.loc37: %.10 = name_ref Op, imports.%import_ref.13 [template = constants.%.11]
// CHECK:STDOUT:   %a.ref.loc41: ref %C = name_ref a, %a
// CHECK:STDOUT:   %Op.ref.loc41: %.13 = name_ref Op, imports.%import_ref.18 [template = constants.%.14]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.3(constants.%Self.3: %AddAssign.type) {
// CHECK:STDOUT:   %Self: %AddAssign.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.3.%Self (%Self.3) [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: @Op.3.%.2 (%.9)](%other: @Op.3.%Self (%Self.3));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.4(constants.%Self.4: %Inc.type) {
// CHECK:STDOUT:   %Self: %Inc.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.4.%Self (%Self.4) [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: @Op.4.%.2 (%.12)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.3(constants.%Self.3) {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.4(constants.%Self.4) {
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:

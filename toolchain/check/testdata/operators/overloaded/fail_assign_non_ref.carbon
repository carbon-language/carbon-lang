// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_assign_non_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_assign_non_ref.carbon

package User;

class C {};

impl C as Core.Inc {
  fn Op[addr self: C*]();
}
impl C as Core.AddAssign {
  fn Op[addr self: C*](other: C);
}

fn TestIncNonRef(a: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+7]]:3: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-10]]:14: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Op[addr self: C*]();
  // CHECK:STDERR:              ^~~~
  // CHECK:STDERR:
  ++a;
}

fn TestAddAssignNonRef(a: C, b: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+6]]:5: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-18]]:14: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:              ^~~~
  a += b;
}

// CHECK:STDOUT: --- fail_assign_non_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Inc [template]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.1, @Inc[%Self.1] [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %Self.1 [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.2, %Op.type.2 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.2 [symbolic]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %.9: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.2: %.9 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.5: type = fn_type @Op.3, @AddAssign[%Self.2] [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %.9, %Op.type.5 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Op.5: %Op.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.type.6: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.6: %Op.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Op.6) [template]
// CHECK:STDOUT:   %TestIncNonRef.type: type = fn_type @TestIncNonRef [template]
// CHECK:STDOUT:   %TestIncNonRef: %TestIncNonRef.type = struct_value () [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.15: %.5 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Op.type.7: type = fn_type @Op.1, @Inc[%C] [template]
// CHECK:STDOUT:   %TestAddAssignNonRef.type: type = fn_type @TestAddAssignNonRef [template]
// CHECK:STDOUT:   %TestAddAssignNonRef: %TestAddAssignNonRef.type = struct_value () [template]
// CHECK:STDOUT:   %.16: %.11 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %Op.type.8: type = fn_type @Op.3, @AddAssign[%C] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Inc = %import_ref.1
// CHECK:STDOUT:     .AddAssign = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+67, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+81, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.4: @Inc.%.1 (%.5) = import_ref Core//prelude/operators/arithmetic, inst+87, loaded [symbolic = @Inc.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.5: @Inc.%Op.type (%Op.type.2) = import_ref Core//prelude/operators/arithmetic, inst+81, loaded [symbolic = @Inc.%Op (constants.%Op.1)]
// CHECK:STDOUT:   %import_ref.6: type = import_ref Core//prelude/operators/arithmetic, inst+35, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/arithmetic, inst+37, unloaded
// CHECK:STDOUT:   %import_ref.9: @AddAssign.%.1 (%.11) = import_ref Core//prelude/operators/arithmetic, inst+61, loaded [symbolic = @AddAssign.%.2 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.10: @AddAssign.%Op.type (%Op.type.5) = import_ref Core//prelude/operators/arithmetic, inst+55, loaded [symbolic = @AddAssign.%Op (constants.%Op.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/arithmetic, inst+81, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestIncNonRef = %TestIncNonRef.decl
// CHECK:STDOUT:     .TestAddAssignNonRef = %TestAddAssignNonRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc15: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc18: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%import_ref.6 [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestIncNonRef.decl: %TestIncNonRef.type = fn_decl @TestIncNonRef [template = constants.%TestIncNonRef] {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc22_18.1: %C = param a
// CHECK:STDOUT:     @TestIncNonRef.%a: %C = bind_name a, %a.loc22_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAddAssignNonRef.decl: %TestAddAssignNonRef.type = fn_decl @TestAddAssignNonRef [template = constants.%TestAddAssignNonRef] {
// CHECK:STDOUT:     %C.ref.loc33_27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc33_24.1: %C = param a
// CHECK:STDOUT:     @TestAddAssignNonRef.%a: %C = bind_name a, %a.loc33_24.1
// CHECK:STDOUT:     %C.ref.loc33_33: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc33_30.1: %C = param b
// CHECK:STDOUT:     @TestAddAssignNonRef.%b: %C = bind_name b, %b.loc33_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc[constants.%Self.1: %.2] {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Inc[%Self] [symbolic = %Op.type (constants.%Op.type.2)]
// CHECK:STDOUT:   %Op: @Inc.%Op.type (%Op.type.2) = struct_value () [symbolic = %Op (constants.%Op.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.2, @Inc.%Op.type (%Op.type.2) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @Inc.%.1 (%.5) = assoc_entity element0, imports.%import_ref.2 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign[constants.%Self.2: %.9] {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.3, @AddAssign[%Self] [symbolic = %Op.type (constants.%Op.type.5)]
// CHECK:STDOUT:   %Op: @AddAssign.%Op.type (%Op.type.5) = struct_value () [symbolic = %Op (constants.%Op.5)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.9, @AddAssign.%Op.type (%Op.type.5) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @AddAssign.%.1 (%.11) = assoc_entity element0, imports.%import_ref.7 [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   .Op = imports.%import_ref.9
// CHECK:STDOUT:   witness = (imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %C as %.2 {
// CHECK:STDOUT:   %Op.decl: %Op.type.3 = fn_decl @Op.2 [template = constants.%Op.3] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc16_21: type = ptr_type %C [template = constants.%.7]
// CHECK:STDOUT:     %self.loc16_14.1: %.7 = param self
// CHECK:STDOUT:     %self.loc16_14.3: %.7 = bind_name self, %self.loc16_14.1
// CHECK:STDOUT:     %.loc16_9: %.7 = addr_pattern %self.loc16_14.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.9 {
// CHECK:STDOUT:   %Op.decl: %Op.type.6 = fn_decl @Op.4 [template = constants.%Op.6] {
// CHECK:STDOUT:     %C.ref.loc19_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc19_21: type = ptr_type %C [template = constants.%.7]
// CHECK:STDOUT:     %self.loc19_14.1: %.7 = param self
// CHECK:STDOUT:     %self.loc19_14.3: %.7 = bind_name self, %self.loc19_14.1
// CHECK:STDOUT:     %.loc19_9: %.7 = addr_pattern %self.loc19_14.3
// CHECK:STDOUT:     %C.ref.loc19_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc19_24.1: %C = param other
// CHECK:STDOUT:     %other.loc19_24.2: %C = bind_name other, %other.loc19_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.13]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.1: %.2) {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.1.%Self (%Self.1) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: %.4]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr @impl.1.%self.loc16_14.3: %.7]();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.3(constants.%Self.2: %.9) {
// CHECK:STDOUT:   %Self: %.9 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.3.%Self (%Self.2) [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: %.10](%other: %Self.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr @impl.2.%self.loc19_14.3: %.7](@impl.2.%other.loc19_24.2: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIncNonRef(%a: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %.1: %Op.type.7 = interface_witness_access @impl.1.%.1, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc30: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Op.call: init %.3 = call %.loc30(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAddAssignNonRef(%a: %C, %b: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.1: %Op.type.8 = interface_witness_access @impl.2.%.1, element0 [template = constants.%Op.6]
// CHECK:STDOUT:   %.loc40: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Op.call: init %.3 = call %.loc40(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inc[constants.%Self.1] {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inc[@Inc.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[constants.%Self.2] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[@AddAssign.%Self] {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inc[constants.%C] {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[constants.%C] {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

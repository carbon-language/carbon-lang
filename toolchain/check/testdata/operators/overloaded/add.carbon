// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/add.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Add {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.AddAssign {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  return a + b;
}

fn TestAssign(a: C*, b: C) {
  *a += b;
}

// CHECK:STDOUT: --- add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Add [template]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.1, @Add[%Self.1] [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.2, %Op.type.2 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.2 [symbolic]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.3: %Op.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %.8: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %Self.2: %.8 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Op.type.5: type = fn_type @Op.3, @AddAssign[%Self.2] [symbolic]
// CHECK:STDOUT:   %.9: type = ptr_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %.8, %Op.type.5 [symbolic]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Op.5: %Op.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.12: type = ptr_type %C [template]
// CHECK:STDOUT:   %Op.type.6: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.6: %Op.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Op.6) [template]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [template]
// CHECK:STDOUT:   %.14: %.4 = assoc_entity element0, imports.%import_ref.11 [symbolic]
// CHECK:STDOUT:   %Op.type.7: type = fn_type @Op.1, @Add[%C] [template]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [template]
// CHECK:STDOUT:   %TestAssign: %TestAssign.type = struct_value () [template]
// CHECK:STDOUT:   %.15: %.10 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %Op.type.8: type = fn_type @Op.3, @AddAssign[%C] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Add = %import_ref.1
// CHECK:STDOUT:     .AddAssign = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Core//prelude/operators/arithmetic, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/arithmetic, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: @Add.%.1 (%.4) = import_ref Core//prelude/operators/arithmetic, inst+29, loaded [symbolic = @Add.%.2 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.5: @Add.%Op.type (%Op.type.2) = import_ref Core//prelude/operators/arithmetic, inst+23, loaded [symbolic = @Add.%Op (constants.%Op.1)]
// CHECK:STDOUT:   %import_ref.6: type = import_ref Core//prelude/operators/arithmetic, inst+35, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//prelude/operators/arithmetic, inst+37, unloaded
// CHECK:STDOUT:   %import_ref.9: @AddAssign.%.1 (%.10) = import_ref Core//prelude/operators/arithmetic, inst+61, loaded [symbolic = @AddAssign.%.2 (constants.%.15)]
// CHECK:STDOUT:   %import_ref.10: @AddAssign.%Op.type (%Op.type.5) = import_ref Core//prelude/operators/arithmetic, inst+55, loaded [symbolic = @AddAssign.%Op (constants.%Op.4)]
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/arithmetic, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//prelude/operators/arithmetic, inst+55, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, imports.%import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, imports.%import_ref.6 [template = constants.%.8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [template = constants.%TestOp] {
// CHECK:STDOUT:     %C.ref.loc26_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc26_11.1: %C = param a
// CHECK:STDOUT:     @TestOp.%a: %C = bind_name a, %a.loc26_11.1
// CHECK:STDOUT:     %C.ref.loc26_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc26_17.1: %C = param b
// CHECK:STDOUT:     @TestOp.%b: %C = bind_name b, %b.loc26_17.1
// CHECK:STDOUT:     %C.ref.loc26_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign.decl: %TestAssign.type = fn_decl @TestAssign [template = constants.%TestAssign] {
// CHECK:STDOUT:     %C.ref.loc30_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc30: type = ptr_type %C [template = constants.%.12]
// CHECK:STDOUT:     %a.loc30_15.1: %.12 = param a
// CHECK:STDOUT:     @TestAssign.%a: %.12 = bind_name a, %a.loc30_15.1
// CHECK:STDOUT:     %C.ref.loc30_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc30_22.1: %C = param b
// CHECK:STDOUT:     @TestAssign.%b: %C = bind_name b, %b.loc30_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add[constants.%Self.1: %.2] {
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.1, @Add[%Self] [symbolic = %Op.type (constants.%Op.type.2)]
// CHECK:STDOUT:   %Op: @Add.%Op.type (%Op.type.2) = struct_value () [symbolic = %Op (constants.%Op.2)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.2, @Add.%Op.type (%Op.type.2) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @Add.%.1 (%.4) = assoc_entity element0, imports.%import_ref.2 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign[constants.%Self.2: %.8] {
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.3, @AddAssign[%Self] [symbolic = %Op.type (constants.%Op.type.5)]
// CHECK:STDOUT:   %Op: @AddAssign.%Op.type (%Op.type.5) = struct_value () [symbolic = %Op (constants.%Op.5)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.8, @AddAssign.%Op.type (%Op.type.5) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %.2: @AddAssign.%.1 (%.10) = assoc_entity element0, imports.%import_ref.7 [symbolic = %.2 (constants.%.11)]
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
// CHECK:STDOUT:     %C.ref.loc18_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc18_9.1: %C = param self
// CHECK:STDOUT:     %self.loc18_9.2: %C = bind_name self, %self.loc18_9.1
// CHECK:STDOUT:     %C.ref.loc18_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc18_18.1: %C = param other
// CHECK:STDOUT:     %other.loc18_18.2: %C = bind_name other, %other.loc18_18.1
// CHECK:STDOUT:     %C.ref.loc18_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C as %.8 {
// CHECK:STDOUT:   %Op.decl: %Op.type.6 = fn_decl @Op.4 [template = constants.%Op.6] {
// CHECK:STDOUT:     %C.ref.loc23_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_21: type = ptr_type %C [template = constants.%.12]
// CHECK:STDOUT:     %self.loc23_14.1: %.12 = param self
// CHECK:STDOUT:     %self.loc23_14.3: %.12 = bind_name self, %self.loc23_14.1
// CHECK:STDOUT:     %.loc23_9: %.12 = addr_pattern %self.loc23_14.3
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc23_24.1: %C = param other
// CHECK:STDOUT:     %other.loc23_24.2: %C = bind_name other, %other.loc23_24.1
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: %Self.1](%other: %Self.1) -> %Self.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[@impl.1.%self.loc18_9.2: %C](@impl.1.%other.loc18_18.2: %C) -> @impl.1.%return.var: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init %C = class_init (), @impl.1.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_14: init %C = converted %.loc19_13.1, %.loc19_13.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc19_14 to @impl.1.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.3(constants.%Self.2: %.8) {
// CHECK:STDOUT:   %Self: %.8 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @Op.3.%Self (%Self.2) [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: %.9](%other: %Self.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr @impl.2.%self.loc23_14.3: %.12](@impl.2.%other.loc23_24.2: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: %C, %b: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.1: %Op.type.7 = interface_witness_access @impl.1.%.1, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc27: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc26: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init %C = call %.loc27(%a.ref, %b.ref) to %.loc26
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a: %.12, %b: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.12 = name_ref a, %a
// CHECK:STDOUT:   %.loc31_3.1: ref %C = deref %a.ref
// CHECK:STDOUT:   %b.ref: %C = name_ref b, %b
// CHECK:STDOUT:   %.1: %Op.type.8 = interface_witness_access @impl.2.%.1, element0 [template = constants.%Op.6]
// CHECK:STDOUT:   %.loc31_6: <bound method> = bound_method %.loc31_3.1, %.1
// CHECK:STDOUT:   %.loc31_3.2: %.12 = addr_of %.loc31_3.1
// CHECK:STDOUT:   %Op.call: init %.3 = call %.loc31_6(%.loc31_3.2, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[constants.%Self.1] {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Add[@Add.%Self] {
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
// CHECK:STDOUT: specific @Add[constants.%C] {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssign[constants.%C] {
// CHECK:STDOUT:   %Self => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/generic.carbon

// --- deduced_type.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

impl forall [T:! type] T as HasF {
  fn F() {}
}

fn G(x: {}) {
  x.(HasF.F)();
}

// --- deduced_type_subst.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F[self: Self]() -> Self;
}

impl forall [T:! type] T as HasF {
  fn F[self: Self]() -> T { return self; }
}

fn G(x: {}) -> {} {
  return x.(HasF.F)();
}

// --- deduced_type_argument.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

class C(T:! type) {}

impl forall [T:! type] C(T) as HasF {
  fn F() {}
}

fn G(x: C({})) {
  x.(HasF.F)();
}

// --- deduced_interface_argument.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F();
}

impl forall [T:! type] {} as HasF(T) {
  fn F() {}
}

fn G(x: {}) {
  x.(HasF({}).F)();
}

// --- fail_incomplete_deduction.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

// TODO: Reject this declaration because U cannot be deduced.
impl forall [T:! type, U:! type] T as HasF {
  fn F() {}
}

fn G(x: {}) {
  // TODO: It'd be nice to include a note here saying that deduction failed because
  // we couldn't deduce a value for 'U'.
  // CHECK:STDERR: fail_incomplete_deduction.carbon:[[@LINE+4]]:3: error: cannot access member of interface `HasF` in type `{}` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   x.(HasF.F)();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  x.(HasF.F)();
}

// --- fail_inconsistent_deduction.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F();
}

impl forall [T:! type] T as HasF(T) {
  fn F() {}
}

class A {}
class B {}

fn G(x: A) {
  // TODO: It'd be nice to include a note here saying that deduction failed because
  // we deduced two different values for `T`.
  // CHECK:STDERR: fail_inconsistent_deduction.carbon:[[@LINE+3]]:3: error: cannot access member of interface `HasF(B)` in type `A` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   x.(HasF(B).F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  x.(HasF(B).F)();
}

// CHECK:STDOUT: --- deduced_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: %HasF.type = facet_value %T, %T [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%.5) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %F.3, @F.2(%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %.5 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.5 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_10.1: %.5 = struct_literal ()
// CHECK:STDOUT:     %.loc12_10.2: type = converted %.loc12_10.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:     %x.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.5 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc8_34.2: <witness> = interface_witness (%F) [symbolic = %.loc8_34.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc8_34.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc8_34.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc8_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc8_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %.5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.5 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @HasF.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_4.1: %F.type.1 = interface_witness_access constants.%.6, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc13_4.2: <specific function> = specific_function %.loc13_4.1, @F.2(constants.%.5) [template = constants.%.7]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc13_4.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc8_34.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.5) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%.5
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc8_34.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_subst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: %HasF.type = facet_value %T, %T [symbolic]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%.6) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %struct: %.6 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %.6 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.6 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.6 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_10.1: %.6 = struct_literal ()
// CHECK:STDOUT:     %.loc12_10.2: type = converted %.loc12_10.1, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:     %.loc12_17.1: %.6 = struct_literal ()
// CHECK:STDOUT:     %.loc12_17.2: type = converted %.loc12_17.1, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:     %x.param: %.6 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.6 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %.6 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%.loc5_14.1 (%.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%.loc5_14.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.1.%.loc5_14.1 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%.loc5_14.1 (%.1) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_14: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_14.2: type = facet_access_type %Self.ref.loc5_14 [symbolic = %.loc5_14.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc5_14.3: type = converted %Self.ref.loc5_14, %.loc5_14.2 [symbolic = %.loc5_14.1 (constants.%.1)]
// CHECK:STDOUT:     %Self.ref.loc5_25: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %.loc5_25.1: type = facet_access_type %Self.ref.loc5_25 [symbolic = %.loc5_14.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc5_25.2: type = converted %Self.ref.loc5_25, %.loc5_25.1 [symbolic = %.loc5_14.1 (constants.%.1)]
// CHECK:STDOUT:     %self.param: @F.1.%.loc5_14.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @F.1.%.loc5_14.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.1.%.loc5_14.1 (%.1) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.1.%.loc5_14.1 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc8_34.2: <witness> = interface_witness (%F) [symbolic = %.loc8_34.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {
// CHECK:STDOUT:       %self.patt: @F.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%T (%T) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @impl.%T.loc8_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.2.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_34.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc8_34.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc8_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.loc5_14.1: type = facet_access_type %Self [symbolic = %.loc5_14.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%.loc5_14.1 (%.1)]() -> @F.1.%.loc5_14.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%T (%T)]() -> @F.2.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%T (%T) = name_ref self, %self
// CHECK:STDOUT:     return %self.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %.6) -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.6 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.2 = name_ref F, @HasF.%.loc5 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_11.1: %F.type.1 = interface_witness_access constants.%.7, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc13_11.2: <bound method> = bound_method %x.ref, %.loc13_11.1
// CHECK:STDOUT:   %.loc13_11.3: <specific function> = specific_function %.loc13_11.2, @F.2(constants.%.6)
// CHECK:STDOUT:   %F.call: init %.6 = call %.loc13_11.3(%x.ref)
// CHECK:STDOUT:   %.loc13_20.1: ref %.6 = temporary_storage
// CHECK:STDOUT:   %.loc13_20.2: ref %.6 = temporary %.loc13_20.1, %F.call
// CHECK:STDOUT:   %struct: %.6 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_22: %.6 = converted %F.call, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc13_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.loc5_14.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc8_34.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.4) {
// CHECK:STDOUT:   %Self => constants.%.4
// CHECK:STDOUT:   %.loc5_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.6) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%.6
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc8_34.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.6) {
// CHECK:STDOUT:   %T => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: %HasF.type = facet_value %C.2, %C.2 [symbolic]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%.3) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%.3) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %.9: <specific function> = specific_function %F.3, @F.2(%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc8_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_9.1, runtime_param<invalid> [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_14.1, runtime_param<invalid> [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_25.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_25.2 (constants.%C.2)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %C.3 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.3 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc14_12: %.3 = struct_literal ()
// CHECK:STDOUT:     %.loc14_10: type = converted %.loc14_12, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%.3) [template = constants.%C.3]
// CHECK:STDOUT:     %x.param: %C.3 = value_param runtime_param0
// CHECK:STDOUT:     %x: %C.3 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc10_14.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc10_25.2: type = class_type @C, @C(%T.loc10_14.2) [symbolic = %C.loc10_25.2 (constants.%C.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc10_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc10_37.2: <witness> = interface_witness (%F) [symbolic = %.loc10_37.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_25.1 as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc10_37.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc10_37.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc10_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.1: type) {
// CHECK:STDOUT:   %T.loc8_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc8: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc10_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %C.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.3 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @HasF.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_4.1: %F.type.1 = interface_witness_access constants.%.8, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc15_4.2: <specific function> = specific_function %.loc15_4.1, @F.2(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc15_4.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc10_14.2) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_25.2 => constants.%C.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc10_37.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc10_14.2) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_25.2 => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%.3) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.3) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%.3
// CHECK:STDOUT:   %C.loc10_25.2 => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc10_37.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_interface_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.1: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %HasF: %HasF.type.1 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.2: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: %HasF.type.2 = facet_value %.3, %.3 [symbolic]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.3: type = facet_type <@HasF, @HasF(%.3)> [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @HasF(%.3) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %HasF.type.3, %F.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.2, @impl(%.3) [template]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.4) [template]
// CHECK:STDOUT:   %.9: <specific function> = specific_function %F.4, @F.2(%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.1 = interface_decl @HasF [template = constants.%HasF] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_25.1: %.3 = struct_literal ()
// CHECK:STDOUT:     %.loc8_25.2: type = converted %.loc8_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_34.1: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_34.2 (constants.%HasF.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %.3 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.3 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_10.1: %.3 = struct_literal ()
// CHECK:STDOUT:     %.loc12_10.2: type = converted %.loc12_10.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %x.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.3 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.2)]
// CHECK:STDOUT:   %Self.2: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc5_9.2: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.2), @HasF.%F.type (%F.type.1) [symbolic = %.loc5_9.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_9.3: @HasF.%.loc5_9.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_9.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.1)] {} {}
// CHECK:STDOUT:     %.loc5_9.1: @HasF.%.loc5_9.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_9.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %HasF.type.loc8_34.2: type = facet_type <@HasF, @HasF(%T.loc8_14.2)> [symbolic = %HasF.type.loc8_34.2 (constants.%HasF.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc8_38.2: <witness> = interface_witness (%F) [symbolic = %.loc8_38.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_34.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc8_38.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc8_38.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc8_38.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc8_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.3 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:   %.loc13_12: %.3 = struct_literal ()
// CHECK:STDOUT:   %.loc13_10: type = converted %.loc13_12, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%.3)> [template = constants.%HasF.type.3]
// CHECK:STDOUT:   %.loc13_14: %.6 = specific_constant @HasF.%.loc5_9.1, @HasF(constants.%.3) [template = constants.%.7]
// CHECK:STDOUT:   %F.ref: %.6 = name_ref F, %.loc13_14 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_4.1: %F.type.3 = interface_witness_access constants.%.8, element0 [template = constants.%F.4]
// CHECK:STDOUT:   %.loc13_4.2: <specific function> = specific_function %.loc13_4.1, @F.2(constants.%.3) [template = constants.%.9]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc13_4.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT:   %.loc5_9.2 => constants.%.1
// CHECK:STDOUT:   %.loc5_9.3 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@impl.%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_34.2 => constants.%HasF.type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc8_38.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%.4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_34.2 => constants.%HasF.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%.3) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc5_9.2 => constants.%.6
// CHECK:STDOUT:   %.loc5_9.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.3) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%.3
// CHECK:STDOUT:   %HasF.type.loc8_34.2 => constants.%HasF.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
// CHECK:STDOUT:   %.loc8_38.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type, %F.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: %HasF.type = facet_value %T, %T [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%HasF.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc9_14.1, runtime_param<invalid> [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc9_24.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc9_24.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc9_24.1, runtime_param<invalid> [symbolic = %U.patt.loc9_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc9_24.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc9_24.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %.5 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %.5 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_10.1: %.5 = struct_literal ()
// CHECK:STDOUT:     %.loc13_10.2: type = converted %.loc13_10.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:     %x.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %x: %.5 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc5: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc5
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type, %U.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc9_24.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc9_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc9_24.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc9_24.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc9_14.2, %U.loc9_24.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc9_44.2: <witness> = interface_witness (%F) [symbolic = %.loc9_44.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc9_44.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc9_44.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc9_44.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc9_14.1: type, @impl.%U.loc9_24.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %.5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %.5 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %.1 = name_ref F, @HasF.%.loc5 [template = constants.%.2]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %U.loc9_24.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc9_24.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc9_44.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2, %U.loc9_24.2) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T
// CHECK:STDOUT:   %U.loc9_24.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc9_24.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.1: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %HasF: %HasF.type.1 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.2: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: %HasF.type.2 = facet_value %T, %T [symbolic]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.3: type = facet_type <@HasF, @HasF(%B)> [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @HasF(%B) [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %HasF.type.3, %F.type.3 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.1 = interface_decl @HasF [template = constants.%HasF] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_24: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_33.1: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_33.2 (constants.%HasF.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %A = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %A = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %x.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %x: %A = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.2)]
// CHECK:STDOUT:   %Self.2: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc5_9.2: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.2), @HasF.%F.type (%F.type.1) [symbolic = %.loc5_9.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc5_9.3: @HasF.%.loc5_9.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_9.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.1)] {} {}
// CHECK:STDOUT:     %.loc5_9.1: @HasF.%.loc5_9.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc5_9.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %HasF.type.loc8_33.2: type = facet_type <@HasF, @HasF(%T.loc8_14.2)> [symbolic = %HasF.type.loc8_33.2 (constants.%HasF.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc8_37.2: <witness> = interface_witness (%F) [symbolic = %.loc8_37.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc8_24 as %HasF.type.loc8_33.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {} {}
// CHECK:STDOUT:     %.loc8_37.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc8_37.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc8_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.2)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc8_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %A = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [template = constants.%HasF.type.3]
// CHECK:STDOUT:   %.loc21: %.8 = specific_constant @HasF.%.loc5_9.1, @HasF(constants.%B) [template = constants.%.9]
// CHECK:STDOUT:   %F.ref: %.8 = name_ref F, %.loc21 [template = constants.%.9]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F.1
// CHECK:STDOUT:   %.loc5_9.2 => constants.%.1
// CHECK:STDOUT:   %.loc5_9.3 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@impl.%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_33.2 => constants.%HasF.type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc8_37.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_33.2 => constants.%HasF.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc5_9.2 => constants.%.8
// CHECK:STDOUT:   %.loc5_9.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:

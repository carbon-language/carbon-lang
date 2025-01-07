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
// CHECK:STDOUT:   %F.type.16de: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.a207: %F.type.16de = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16de [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.c722: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.6cdc: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.0740: %F.type.6cdc = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %T, %impl_witness.c722 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.93bd: <witness> = impl_witness (@impl.%F.decl), @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.type.e4d7: type = fn_type @F.2, @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.da17: %F.type.e4d7 = struct_value () [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.da17, @F.2(%empty_struct_type) [template]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.c722)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_struct_type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.16de = fn_decl @F.1 [template = constants.%F.a207] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.c722)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.6cdc)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.6cdc) = struct_value () [symbolic = %F (constants.%F.0740)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.6cdc) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.0740)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
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
// CHECK:STDOUT: fn @G(%x.param_patt: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %F.type.16de = impl_witness_access constants.%impl_witness.93bd, element0 [template = constants.%F.da17]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%empty_struct_type) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.c722
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.6cdc
// CHECK:STDOUT:   %F => constants.%F.0740
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.93bd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.e4d7
// CHECK:STDOUT:   %F => constants.%F.da17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_subst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.16de: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.a207: %F.type.16de = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16de [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.11b3: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.6cdc: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.0740: %F.type.6cdc = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %T, %impl_witness.11b3 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.f452: <witness> = impl_witness (@impl.%F.decl), @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.type.e4d7: type = fn_type @F.2, @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.da17: %F.type.e4d7 = struct_value () [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.11b3)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_struct_type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_17.2: type = converted %.loc12_17.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.16de = fn_decl @F.1 [template = constants.%F.a207] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_25: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_25: type = facet_access_type %Self.ref.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5_25: type = converted %Self.ref.loc5_25, %Self.as_type.loc5_25 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_14: %HasF.type = name_ref Self, @HasF.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref.loc5_14 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref.loc5_14, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.11b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.6cdc)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.6cdc) = struct_value () [symbolic = %F (constants.%F.0740)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.6cdc) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.0740)] {
// CHECK:STDOUT:       %self.patt: @F.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%T (%T) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @impl.%T.loc8_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.2.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @F.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%Self: %HasF.type) {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)]() -> @F.1.%Self.as_type.loc5_14.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%T (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%T (%T)]() -> @F.2.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%T (%T) = name_ref self, %self
// CHECK:STDOUT:     return %self.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %F.type.16de = impl_witness_access constants.%impl_witness.f452, element0 [template = constants.%F.da17]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.bound, @F.2(constants.%empty_struct_type)
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.specific_fn(%x.ref)
// CHECK:STDOUT:   %.loc13_21.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc13_21.2: ref %empty_struct_type = temporary %.loc13_21.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_22: %empty_struct_type = converted %F.call, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc13_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.11b3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.6cdc
// CHECK:STDOUT:   %F => constants.%F.0740
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.f452
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.e4d7
// CHECK:STDOUT:   %F => constants.%F.da17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_type_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.16de: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.a207: %F.type.16de = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16de [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %impl_witness.cf52: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.8231: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.7438: %F.type.8231 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %C.37e2, %impl_witness.cf52 [symbolic]
// CHECK:STDOUT:   %C.27b8: type = class_type @C, @C(%empty_struct_type) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %impl_witness.3d01: <witness> = impl_witness (@impl.%F.decl), @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.type.239b: type = fn_type @F.2, @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.c564: %F.type.239b = struct_value () [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.c564, @F.2(%empty_struct_type) [template]
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
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
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
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc10_27.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc10_27.2 (constants.%C.37e2)]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.cf52)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %C.27b8 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.27b8 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.27b8 = value_param runtime_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %C [template = constants.%C.27b8] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc14_13.2: type = converted %.loc14_12, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%empty_struct_type) [template = constants.%C.27b8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.27b8 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.16de = fn_decl @F.1 [template = constants.%F.a207] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc10_14.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc10_27.2: type = class_type @C, @C(%T.loc10_14.2) [symbolic = %C.loc10_27.2 (constants.%C.37e2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc10_14.2) [symbolic = %impl_witness (constants.%impl_witness.cf52)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc10_14.2) [symbolic = %F.type (constants.%F.type.8231)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.8231) = struct_value () [symbolic = %F (constants.%F.7438)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.loc10_27.1 as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.8231) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.7438)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e2
// CHECK:STDOUT:     complete_type_witness = %complete_type
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
// CHECK:STDOUT: fn @G(%x.param_patt: %C.27b8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %C.27b8 = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %F.type.16de = impl_witness_access constants.%impl_witness.3d01, element0 [template = constants.%F.c564]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%empty_struct_type) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
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
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %C.loc10_27.2 => constants.%C.37e2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.cf52
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.8231
// CHECK:STDOUT:   %F => constants.%F.7438
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@impl.%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc10_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %C.loc10_27.2 => constants.%C.27b8
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3d01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.239b
// CHECK:STDOUT:   %F => constants.%F.c564
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced_interface_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.fe31: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe31 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.a193: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %HasF.type.a193 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.38a5: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.81b2: %F.type.38a5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.cd16: type = assoc_entity_type %HasF.type.a193, %F.type.38a5 [symbolic]
// CHECK:STDOUT:   %assoc0.1d6c: %F.assoc_type.cd16 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.a193 [symbolic]
// CHECK:STDOUT:   %impl_witness.b356: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.7da6: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.421d: %F.type.7da6 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.a193 = facet_value %empty_struct_type, %impl_witness.b356 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.914b: type = facet_type <@HasF, @HasF(%empty_struct_type)> [template]
// CHECK:STDOUT:   %F.type.6ab0: type = fn_type @F.1, @HasF(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.5c75: %F.type.6ab0 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.26be: type = assoc_entity_type %HasF.type.914b, %F.type.6ab0 [template]
// CHECK:STDOUT:   %assoc0.5895: %F.assoc_type.26be = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %HasF.type.914b [template]
// CHECK:STDOUT:   %impl_witness.7cdf: <witness> = impl_witness (@impl.%F.decl), @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.type.c456: type = fn_type @F.2, @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.b2ae: %F.type.c456 = struct_value () [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.b2ae, @F.2(%empty_struct_type) [template]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe31 = interface_decl @HasF [template = constants.%HasF.generic] {
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
// CHECK:STDOUT:     %.loc8_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_25.2: type = converted %.loc8_25.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe31 = name_ref HasF, file.%HasF.decl [template = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_36.1: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_36.2 (constants.%HasF.type.a193)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness.b356)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_struct_type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_10.1: type = splice_block %.loc12_10.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc12_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc12_10.3: type = converted %.loc12_10.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.a193)]
// CHECK:STDOUT:   %Self.2: %HasF.type.a193 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.38a5)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.38a5) = struct_value () [symbolic = %F (constants.%F.81b2)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.a193), @HasF.%F.type (%F.type.38a5) [symbolic = %F.assoc_type (constants.%F.assoc_type.cd16)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @HasF.%F.assoc_type (%F.assoc_type.cd16) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.1d6c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.a193) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.38a5) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.81b2)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @HasF.%F.assoc_type (%F.assoc_type.cd16) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.1d6c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %HasF.type.loc8_36.2: type = facet_type <@HasF, @HasF(%T.loc8_14.2)> [symbolic = %HasF.type.loc8_36.2 (constants.%HasF.type.a193)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%HasF.type.loc8_36.2 (%HasF.type.a193) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.b356)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.7da6)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.7da6) = struct_value () [symbolic = %F (constants.%F.421d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_25.2 as %HasF.type.loc8_36.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.7da6) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.421d)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.a193)) {
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
// CHECK:STDOUT: fn @G(%x.param_patt: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: %HasF.type.fe31 = name_ref HasF, file.%HasF.decl [template = constants.%HasF.generic]
// CHECK:STDOUT:   %.loc13_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_13: type = converted %.loc13_12, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%empty_struct_type)> [template = constants.%HasF.type.914b]
// CHECK:STDOUT:   %.loc13_14: %F.assoc_type.26be = specific_constant @HasF.%assoc0.loc5_9.1, @HasF(constants.%empty_struct_type) [template = constants.%assoc0.5895]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.26be = name_ref F, %.loc13_14 [template = constants.%assoc0.5895]
// CHECK:STDOUT:   %impl.elem0: %F.type.6ab0 = impl_witness_access constants.%impl_witness.7cdf, element0 [template = constants.%F.b2ae]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%empty_struct_type) [template = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.a193
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.38a5
// CHECK:STDOUT:   %F => constants.%F.81b2
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.cd16
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.1d6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_36.2 => constants.%HasF.type.a193
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b356
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.7da6
// CHECK:STDOUT:   %F => constants.%F.421d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@impl.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.914b
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.6ab0
// CHECK:STDOUT:   %F => constants.%F.5c75
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.26be
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.5895
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %HasF.type.loc8_36.2 => constants.%HasF.type.914b
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.7cdf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.c456
// CHECK:STDOUT:   %F => constants.%F.b2ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [template]
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.16de: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.a207: %F.type.16de = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %HasF.type, %F.type.16de [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %F.type.c6f6: type = fn_type @F.2, @impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %F.5548: %F.type.c6f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type = facet_value %T, %impl_witness [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T, constants.%U) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %empty_struct_type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_struct_type = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_struct_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_10.1: type = splice_block %.loc13_10.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc13_10.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc13_10.3: type = converted %.loc13_10.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_struct_type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: %HasF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.16de = fn_decl @F.1 [template = constants.%F.a207] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc9_14.1: type, %U.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc9_24.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc9_24.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc9_24.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc9_24.2 (constants.%U.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc9_14.2, %U.loc9_24.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc9_14.2, %U.loc9_24.2) [symbolic = %F.type (constants.%F.type.c6f6)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.c6f6) = struct_value () [symbolic = %F (constants.%F.5548)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %HasF.ref {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.c6f6) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.5548)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
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
// CHECK:STDOUT: fn @G(%x.param_patt: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %empty_struct_type = name_ref x, %x
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%HasF.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @HasF.%assoc0 [template = constants.%assoc0]
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
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.c6f6
// CHECK:STDOUT:   %F => constants.%F.5548
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc9_14.2, %U.loc9_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T, constants.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inconsistent_deduction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.fe31: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe31 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.a193: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %HasF.type.a193 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.38a5: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.81b2: %F.type.38a5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.cd16: type = assoc_entity_type %HasF.type.a193, %F.type.38a5 [symbolic]
// CHECK:STDOUT:   %assoc0.1d6c: %F.assoc_type.cd16 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %HasF.type.a193 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.type.7a5b: type = fn_type @F.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %F.ddef: %F.type.7a5b = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.a193 = facet_value %T, %impl_witness [symbolic]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.0866: type = facet_type <@HasF, @HasF(%B)> [template]
// CHECK:STDOUT:   %F.type.d904: type = fn_type @F.1, @HasF(%B) [template]
// CHECK:STDOUT:   %F.765b: %F.type.d904 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.cee1: type = assoc_entity_type %HasF.type.0866, %F.type.d904 [template]
// CHECK:STDOUT:   %assoc0.f482: %F.assoc_type.cee1 = assoc_entity element0, @HasF.%F.decl [template]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe31 = interface_decl @HasF [template = constants.%HasF.generic] {
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
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe31 = name_ref HasF, file.%HasF.decl [template = constants.%HasF.generic]
// CHECK:STDOUT:     %T.ref.loc8_34: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %HasF.type.loc8_35.1: type = facet_type <@HasF, @HasF(constants.%T)> [symbolic = %HasF.type.loc8_35.2 (constants.%HasF.type.a193)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %A = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %A = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %x: %A = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.a193)]
// CHECK:STDOUT:   %Self.2: %HasF.type.a193 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.38a5)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.38a5) = struct_value () [symbolic = %F (constants.%F.81b2)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.a193), @HasF.%F.type (%F.type.38a5) [symbolic = %F.assoc_type (constants.%F.assoc_type.cd16)]
// CHECK:STDOUT:   %assoc0.loc5_9.2: @HasF.%F.assoc_type (%F.assoc_type.cd16) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.1d6c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.a193) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.38a5) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.81b2)] {} {}
// CHECK:STDOUT:     %assoc0.loc5_9.1: @HasF.%F.assoc_type (%F.assoc_type.cd16) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_9.2 (constants.%assoc0.1d6c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_9.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %HasF.type.loc8_35.2: type = facet_type <@HasF, @HasF(%T.loc8_14.2)> [symbolic = %HasF.type.loc8_35.2 (constants.%HasF.type.a193)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%HasF.type.loc8_35.2 (%HasF.type.a193) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%T.loc8_14.2) [symbolic = %F.type (constants.%F.type.7a5b)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.7a5b) = struct_value () [symbolic = %F (constants.%F.ddef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc8_24 as %HasF.type.loc8_35.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.7a5b) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.ddef)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.a193)) {
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
// CHECK:STDOUT:   %HasF.ref: %HasF.type.fe31 = name_ref HasF, file.%HasF.decl [template = constants.%HasF.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(constants.%B)> [template = constants.%HasF.type.0866]
// CHECK:STDOUT:   %.loc21: %F.assoc_type.cee1 = specific_constant @HasF.%assoc0.loc5_9.1, @HasF(constants.%B) [template = constants.%assoc0.f482]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.cee1 = name_ref F, %.loc21 [template = constants.%assoc0.f482]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.a193
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.38a5
// CHECK:STDOUT:   %F => constants.%F.81b2
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.cd16
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.1d6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %HasF.type.loc8_35.2 => constants.%HasF.type.a193
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.7a5b
// CHECK:STDOUT:   %F => constants.%F.ddef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@impl.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%HasF.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%B) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%B
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.0866
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.d904
// CHECK:STDOUT:   %F => constants.%F.765b
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.cee1
// CHECK:STDOUT:   %assoc0.loc5_9.2 => constants.%assoc0.f482
// CHECK:STDOUT: }
// CHECK:STDOUT:

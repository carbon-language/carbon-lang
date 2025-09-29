// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/const.carbon

// --- add_const.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> X;
let value: X = Init();
var reference: X;
let ptr: X* = &reference;

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  var i: const X = Init() as const X;
  let v: const X = value as const X;
  let a: const X* = &(reference as const X);
  let b: const X* = ptr as const X*;
  //@dump-sem-ir-end
}

// --- remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> const X;
let value: const X = Init();

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  var i: X = Init() as X;
  let v: X = value as X;
  //@dump-sem-ir-end
}

// --- fail_cannot_remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

var reference: const X;
let ptr: const X* = &reference;

fn Use() {
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+7]]:17: error: cannot convert expression of type `const X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:17: note: type `const X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `const X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:15: note: type `const X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR:
  let b: X* = ptr as X*;
}

// --- unsafe_remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

var reference: const X;
let ptr: const X* = &reference;

fn Use() {
  //@dump-sem-ir-begin
  let a: X* = &(reference unsafe as X);
  let b: X* = ptr unsafe as X*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %const.dde: type = const_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.d7f91: type = pattern_type %const.dde [concrete]
// CHECK:STDOUT:   %ptr.cbd: type = ptr_type %const.dde [concrete]
// CHECK:STDOUT:   %pattern_type.855: type = pattern_type %ptr.cbd [concrete]
// CHECK:STDOUT:   %reference.var: ref %const.dde = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.0c5: %ptr.cbd = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %AggregateT: %type_where = bind_symbolic_name AggregateT, 0 [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.190: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%AggregateT) [symbolic]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.8a0: %AggregateT.as_type.as.Destroy.impl.Op.type.190 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.079: <witness> = impl_witness imports.%Destroy.impl_witness_table.2d3, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.facet.074: %Destroy.type = facet_value %X, (%Destroy.impl_witness.079) [concrete]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.type.4e9: type = fn_type @const.as.Destroy.impl.Op, @const.as.Destroy.impl(%Destroy.facet.074) [concrete]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.3cc: %const.as.Destroy.impl.Op.type.4e9 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d51: @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op.type (%AggregateT.as_type.as.Destroy.impl.Op.type.190) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @AggregateT.as_type.as.Destroy.impl.%AggregateT.as_type.as.Destroy.impl.Op (constants.%AggregateT.as_type.as.Destroy.impl.Op.8a0)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.2d3 = impl_witness_table (%Core.import_ref.d51), @AggregateT.as_type.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.d7f91 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.d7f91 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %const.dde = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc14_3.1: ref %const.dde = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X = call %Init.ref() to %.loc14_3.1
// CHECK:STDOUT:   %X.ref.loc14_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc14_30: type = const_type %X.ref.loc14_36 [concrete = constants.%const.dde]
// CHECK:STDOUT:   %.loc14_27.1: init %const.dde = as_compatible %Init.call
// CHECK:STDOUT:   %.loc14_27.2: init %const.dde = converted %Init.call, %.loc14_27.1
// CHECK:STDOUT:   assign %i.var, %.loc14_27.2
// CHECK:STDOUT:   %.loc14_10: type = splice_block %const.loc14_10 [concrete = constants.%const.dde] {
// CHECK:STDOUT:     %X.ref.loc14_16: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc14_10: type = const_type %X.ref.loc14_16 [concrete = constants.%const.dde]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %const.dde = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.d7f91 = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc15_35: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc15_29: type = const_type %X.ref.loc15_35 [concrete = constants.%const.dde]
// CHECK:STDOUT:   %.loc15_26.1: %const.dde = as_compatible %value.ref
// CHECK:STDOUT:   %.loc15_26.2: %const.dde = converted %value.ref, %.loc15_26.1
// CHECK:STDOUT:   %.loc15_10: type = splice_block %const.loc15_10 [concrete = constants.%const.dde] {
// CHECK:STDOUT:     %X.ref.loc15_16: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc15_10: type = const_type %X.ref.loc15_16 [concrete = constants.%const.dde]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %const.dde = bind_name v, %.loc15_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.855 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc16_42: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc16_36: type = const_type %X.ref.loc16_42 [concrete = constants.%const.dde]
// CHECK:STDOUT:   %.loc16_33.1: ref %const.dde = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc16_33.2: ref %const.dde = converted %reference.ref, %.loc16_33.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr.loc16: %ptr.cbd = addr_of %.loc16_33.2 [concrete = constants.%addr.0c5]
// CHECK:STDOUT:   %.loc16_17: type = splice_block %ptr.loc16 [concrete = constants.%ptr.cbd] {
// CHECK:STDOUT:     %X.ref.loc16_16: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc16_10: type = const_type %X.ref.loc16_16 [concrete = constants.%const.dde]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %const.loc16_10 [concrete = constants.%ptr.cbd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.cbd = bind_name a, %addr.loc16
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.855 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.d17 = name_ref ptr, file.%ptr.loc9_5
// CHECK:STDOUT:   %X.ref.loc17_34: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc17_28: type = const_type %X.ref.loc17_34 [concrete = constants.%const.dde]
// CHECK:STDOUT:   %ptr.loc17_35: type = ptr_type %const.loc17_28 [concrete = constants.%ptr.cbd]
// CHECK:STDOUT:   %.loc17_25.1: %ptr.cbd = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc17_25.2: %ptr.cbd = converted %ptr.ref, %.loc17_25.1
// CHECK:STDOUT:   %.loc17_17: type = splice_block %ptr.loc17_17 [concrete = constants.%ptr.cbd] {
// CHECK:STDOUT:     %X.ref.loc17_16: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc17_10: type = const_type %X.ref.loc17_16 [concrete = constants.%const.dde]
// CHECK:STDOUT:     %ptr.loc17_17: type = ptr_type %const.loc17_10 [concrete = constants.%ptr.cbd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.cbd = bind_name b, %.loc17_25.2
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc14_3.2: %type_where = converted constants.%X, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value constants.%X, (constants.%Destroy.impl_witness.079) [concrete = constants.%Destroy.facet.074]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%X, %Destroy.facet [concrete = constants.%Destroy.facet.074]
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%const.as.Destroy.impl.Op.3cc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %i.var, %const.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc14: %ptr.cbd = addr_of %i.var
// CHECK:STDOUT:   %const.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc14)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- remove_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const.dde: type = const_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.dc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.177: %AggregateT.as_type.as.Destroy.impl.Op.type.dc1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.019 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.019 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc12_3.1: ref %X = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %const.dde = call %Init.ref() to %.loc12_3.1
// CHECK:STDOUT:   %X.ref.loc12_24: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_21.1: init %X = as_compatible %Init.call
// CHECK:STDOUT:   %.loc12_21.2: init %X = converted %Init.call, %.loc12_21.1
// CHECK:STDOUT:   assign %i.var, %.loc12_21.2
// CHECK:STDOUT:   %X.ref.loc12_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.019 = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %const.dde = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc13_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_20.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc13_20.2: %X = converted %value.ref, %.loc13_20.1
// CHECK:STDOUT:   %X.ref.loc13_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = bind_name v, %.loc13_20.2
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc12_3.2: %type_where = converted constants.%X, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %i.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %i.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.cbd: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.373: %ptr.d17 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1c6 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %const = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc11_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_34.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc11_34.2: ref %X = converted %reference.ref, %.loc11_34.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %.loc11_34.2 [concrete = constants.%addr.373]
// CHECK:STDOUT:   %.loc11_11: type = splice_block %ptr.loc11 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc11_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc11: type = ptr_type %X.ref.loc11_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.d17 = bind_name a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.1c6 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.cbd = name_ref ptr, file.%ptr.loc7_5
// CHECK:STDOUT:   %X.ref.loc12_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc12_30: type = ptr_type %X.ref.loc12_29 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   %.loc12_26.1: %ptr.d17 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc12_26.2: %ptr.d17 = converted %ptr.ref, %.loc12_26.1
// CHECK:STDOUT:   %.loc12_11: type = splice_block %ptr.loc12_11 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc12_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc12_11: type = ptr_type %X.ref.loc12_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.d17 = bind_name b, %.loc12_26.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/min_prelude/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/min_prelude/tuple.carbon

// --- test.carbon

library "[[@TEST_NAME]]";

class X {
  // ...
}

fn Make() -> X;

fn Let() {
  // This should create value bindings for both tuple elements.
  let a: (X, X) = (Make(), Make()) as (X, X);
}

fn Var() {
  // This should initialize both tuple elements in place.
  var b: (X, X) = (Make(), Make()) as (X, X);
}

// CHECK:STDOUT: --- test.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Let.type: type = fn_type @Let [concrete]
// CHECK:STDOUT:   %Let: %Let.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.b67: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.599: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%X) [concrete]
// CHECK:STDOUT:   %Op.type.9e1: type = fn_type @Op.2, @impl(%X) [concrete]
// CHECK:STDOUT:   %Op.33a: %Op.type.9e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.177: %Destroy.type = facet_value %X, (%Destroy.impl_witness.599) [concrete]
// CHECK:STDOUT:   %.bfc: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.177 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.d1f: <specific function> = specific_function %Op.33a, @Op.2(%X) [concrete]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [concrete]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.543: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%tuple.type.b67) [concrete]
// CHECK:STDOUT:   %Op.type.df3: type = fn_type @Op.2, @impl(%tuple.type.b67) [concrete]
// CHECK:STDOUT:   %Op.279: %Op.type.df3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.b75: <witness> = complete_type_witness %tuple.type.b67 [concrete]
// CHECK:STDOUT:   %Destroy.facet.54c: %Destroy.type = facet_value %tuple.type.b67, (%Destroy.impl_witness.543) [concrete]
// CHECK:STDOUT:   %.fdd: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.54c [concrete]
// CHECK:STDOUT:   %Op.specific_fn.837: <specific function> = specific_function %Op.279, @Op.2(%tuple.type.b67) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Let = %Let.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Let.decl: %Let.type = fn_decl @Let [concrete = constants.%Let] {} {}
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [concrete = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.b67 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc12_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_25.1: ref %X = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_25: %.bfc = impl_witness_access constants.%Destroy.impl_witness.599, element0 [concrete = constants.%Op.33a]
// CHECK:STDOUT:   %bound_method.loc12_25: <bound method> = bound_method %.loc12_25.1, %impl.elem0.loc12_25
// CHECK:STDOUT:   %Make.call.loc12_25: init %X = call %Make.ref.loc12_20() to %.loc12_25.1
// CHECK:STDOUT:   %Make.ref.loc12_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc12_33: %.bfc = impl_witness_access constants.%Destroy.impl_witness.599, element0 [concrete = constants.%Op.33a]
// CHECK:STDOUT:   %bound_method.loc12_33: <bound method> = bound_method %.loc12_33.1, %impl.elem0.loc12_33
// CHECK:STDOUT:   %Make.call.loc12_33: init %X = call %Make.ref.loc12_28() to %.loc12_33.1
// CHECK:STDOUT:   %.loc12_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc12_25, %Make.call.loc12_33)
// CHECK:STDOUT:   %X.ref.loc12_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc12_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc12_40, %X.ref.loc12_43)
// CHECK:STDOUT:   %.loc12_44.2: type = converted %.loc12_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc12_15.1: type = splice_block %.loc12_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc12_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc12_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc12_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc12_11, %X.ref.loc12_14)
// CHECK:STDOUT:     %.loc12_15.3: type = converted %.loc12_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_25.2: ref %X = temporary %.loc12_25.1, %Make.call.loc12_25
// CHECK:STDOUT:   %.loc12_25.3: %X = bind_value %.loc12_25.2
// CHECK:STDOUT:   %.loc12_33.2: ref %X = temporary %.loc12_33.1, %Make.call.loc12_33
// CHECK:STDOUT:   %.loc12_33.3: %X = bind_value %.loc12_33.2
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc12_25.3, %.loc12_33.3)
// CHECK:STDOUT:   %.loc12_34.2: %tuple.type.b67 = converted %.loc12_34.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.b67 = bind_name a, %.loc12_34.2
// CHECK:STDOUT:   %specific_fn.loc12_33: <specific function> = specific_function %impl.elem0.loc12_33, @Op.2(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc12_33.1, %specific_fn.loc12_33
// CHECK:STDOUT:   %.loc12_33.4: %X = bind_value %.loc12_33.1
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc12_33.4)
// CHECK:STDOUT:   %specific_fn.loc12_25: <specific function> = specific_function %impl.elem0.loc12_25, @Op.2(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc12_25.1, %specific_fn.loc12_25
// CHECK:STDOUT:   %.loc12_25.4: %X = bind_value %.loc12_25.1
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc12_25.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.b67 = binding_pattern b
// CHECK:STDOUT:     %.loc17_3.1: %tuple.type.b67 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.b67 = var b
// CHECK:STDOUT:   %impl.elem0.loc17_3: %.fdd = impl_witness_access constants.%Destroy.impl_witness.543, element0 [concrete = constants.%Op.279]
// CHECK:STDOUT:   %bound_method.loc17_3: <bound method> = bound_method %b.var, %impl.elem0.loc17_3
// CHECK:STDOUT:   %Make.ref.loc17_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0.loc17_34: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %impl.elem0.loc17_25: %.bfc = impl_witness_access constants.%Destroy.impl_witness.599, element0 [concrete = constants.%Op.33a]
// CHECK:STDOUT:   %bound_method.loc17_25: <bound method> = bound_method %tuple.elem0.loc17_34, %impl.elem0.loc17_25
// CHECK:STDOUT:   %Make.call.loc17_25: init %X = call %Make.ref.loc17_20() to %tuple.elem0.loc17_34
// CHECK:STDOUT:   %Make.ref.loc17_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1.loc17_34: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %impl.elem0.loc17_33: %.bfc = impl_witness_access constants.%Destroy.impl_witness.599, element0 [concrete = constants.%Op.33a]
// CHECK:STDOUT:   %bound_method.loc17_33: <bound method> = bound_method %tuple.elem1.loc17_34, %impl.elem0.loc17_33
// CHECK:STDOUT:   %Make.call.loc17_33: init %X = call %Make.ref.loc17_28() to %tuple.elem1.loc17_34
// CHECK:STDOUT:   %.loc17_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc17_25, %Make.call.loc17_33)
// CHECK:STDOUT:   %X.ref.loc17_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc17_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc17_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc17_40, %X.ref.loc17_43)
// CHECK:STDOUT:   %.loc17_44.2: type = converted %.loc17_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc17_34.2: init %tuple.type.b67 = tuple_init (%Make.call.loc17_25, %Make.call.loc17_33) to %b.var
// CHECK:STDOUT:   %.loc17_3.2: init %tuple.type.b67 = converted %.loc17_34.1, %.loc17_34.2
// CHECK:STDOUT:   assign %b.var, %.loc17_3.2
// CHECK:STDOUT:   %.loc17_15.1: type = splice_block %.loc17_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc17_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc17_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc17_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc17_11, %X.ref.loc17_14)
// CHECK:STDOUT:     %.loc17_15.3: type = converted %.loc17_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.b67 = bind_name b, %b.var
// CHECK:STDOUT:   %specific_fn.loc17_33: <specific function> = specific_function %impl.elem0.loc17_33, @Op.2(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %tuple.elem1.loc17_34, %specific_fn.loc17_33
// CHECK:STDOUT:   %.loc17_34.3: %X = bind_value %tuple.elem1.loc17_34
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc17_34.3)
// CHECK:STDOUT:   %specific_fn.loc17_25: <specific function> = specific_function %impl.elem0.loc17_25, @Op.2(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %tuple.elem0.loc17_34, %specific_fn.loc17_25
// CHECK:STDOUT:   %.loc17_34.4: %X = bind_value %tuple.elem0.loc17_34
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc17_34.4)
// CHECK:STDOUT:   %specific_fn.loc17_3: <specific function> = specific_function %impl.elem0.loc17_3, @Op.2(constants.%tuple.type.b67) [concrete = constants.%Op.specific_fn.837]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %b.var, %specific_fn.loc17_3
// CHECK:STDOUT:   %tuple.elem0.loc17_3: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc17_3.3: %X = bind_value %tuple.elem0.loc17_3
// CHECK:STDOUT:   %tuple.elem1.loc17_3: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc17_3.4: %X = bind_value %tuple.elem1.loc17_3
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc17_3.3, %.loc17_3.4)
// CHECK:STDOUT:   %.loc17_3.5: %tuple.type.b67 = converted %b.var, %tuple
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc17_3.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.599
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.9e1
// CHECK:STDOUT:   %Op => constants.%Op.33a
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%tuple.type.b67) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b67
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.543
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.df3
// CHECK:STDOUT:   %Op => constants.%Op.279
// CHECK:STDOUT:   %require_complete => constants.%complete_type.b75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%tuple.type.b67) {
// CHECK:STDOUT:   %T => constants.%tuple.type.b67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.439: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.417: %Op.type.439 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, @Destroy.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.b22: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.d6d: %Op.type.b22 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T, (%Destroy.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Destroy = %Destroy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.decl: type = interface_decl @Destroy [concrete = constants.%Destroy.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, file.%Destroy.decl [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type.439 = fn_decl @Op.1 [concrete = constants.%Op.417] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_15.1: type = splice_block %.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Destroy.type = name_ref Self, @Destroy.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_15.2: type = converted %Self.ref, %Self.as_type.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T.loc11_14.2) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc11_14.2) [symbolic = %Op.type (constants.%Op.type.b22)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.b22) = struct_value () [symbolic = %Op (constants.%Op.d6d)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc11_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Destroy.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.b22) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.d6d)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Destroy.%Self: %Destroy.type) {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Destroy.facet) {
// CHECK:STDOUT:   %Self => constants.%Destroy.facet
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

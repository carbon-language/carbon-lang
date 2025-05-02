// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/struct.carbon

interface Z {
  let X:! type;
  fn ZZ() -> X;
}

class C { adapt (); }
class D { adapt (); }

impl {.a: (), .b: ()} as Z where .X = C {
  fn ZZ() -> C { return () as C; }
}
// Structs with different field names are different types.
impl {.aa: (), .bb: ()} as Z where .X = D {
  fn ZZ() -> D { return () as D; }
}

// Are structs with different field orders different types too, for impl lookup?
// Or should this impl be diagnosed as overlapping with the impl on `{.a, .b}`?
// Raised in: https://github.com/carbon-language/carbon-lang/issues/5413
impl {.b: (), .a: ()} as Z where .X = D {
  fn ZZ() -> D { return () as D; }
}

fn F(T:! Z) -> T.X {
  return T.ZZ();
}

fn G() {
  let c1: C = F({.a: (), .b: ()});
  let d1: D = F({.aa: (), .bb: ()});

  let d2: D = F({.b: (), .a: ()});
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.0f75b1.1: <witness> = lookup_impl_witness %Self, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet.f699e2.1: %Z.type = facet_value %Self.as_type, (%Z.lookup_impl_witness.0f75b1.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4a5fb1.1: type = impl_witness_access %Z.lookup_impl_witness.0f75b1.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c15a81.1: type = pattern_type %impl.elem0.4a5fb1.1 [symbolic]
// CHECK:STDOUT:   %ZZ.type.df1: type = fn_type @ZZ.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ZZ.a9f: %ZZ.type.df1 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, @Z.%ZZ.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.a62: <witness> = lookup_impl_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %Z.facet.2a2: %Z.type = facet_value %.Self.as_type, (%Z.lookup_impl_witness.a62) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7a8: type = impl_witness_access %Z.lookup_impl_witness.a62, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.7c5: type = facet_type <@Z where %impl.elem0.7a8 = %C> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.bd6: <witness> = impl_witness file.%Z.impl_witness_table.loc22 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ZZ.type.3f4: type = fn_type @ZZ.2 [concrete]
// CHECK:STDOUT:   %ZZ.529: %ZZ.type.3f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet.56d: %Z.type = facet_value %struct_type.a.b, (%Z.impl_witness.bd6) [concrete]
// CHECK:STDOUT:   %empty_tuple.af4: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.a69: %C = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.aa.bb: type = struct_type {.aa: %empty_tuple.type, .bb: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Z_where.type.47f: type = facet_type <@Z where %impl.elem0.7a8 = %D> [concrete]
// CHECK:STDOUT:   %Z.impl_witness.988: <witness> = impl_witness file.%Z.impl_witness_table.loc26 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %ZZ.type.4e2: type = fn_type @ZZ.3 [concrete]
// CHECK:STDOUT:   %ZZ.8f1: %ZZ.type.4e2 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet.501: %Z.type = facet_value %struct_type.aa.bb, (%Z.impl_witness.988) [concrete]
// CHECK:STDOUT:   %empty_tuple.bf9: %D = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: %empty_tuple.type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Z.impl_witness.abe: <witness> = impl_witness file.%Z.impl_witness_table.loc33 [concrete]
// CHECK:STDOUT:   %ZZ.type.559: type = fn_type @ZZ.4 [concrete]
// CHECK:STDOUT:   %ZZ.0f1: %ZZ.type.559 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.facet.b63: %Z.type = facet_value %struct_type.b.a, (%Z.impl_witness.abe) [concrete]
// CHECK:STDOUT:   %T: %Z.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.5af: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.0f75b1.2: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet.f699e2.2: %Z.type = facet_value %T.as_type, (%Z.lookup_impl_witness.0f75b1.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4a5fb1.2: type = impl_witness_access %Z.lookup_impl_witness.0f75b1.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c15a81.2: type = pattern_type %impl.elem0.4a5fb1.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %impl.elem0.4a5fb1.2 [symbolic]
// CHECK:STDOUT:   %.881: type = fn_type_with_self_type %ZZ.type.df1, %Z.facet.f699e2.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.881 = impl_witness_access %Z.lookup_impl_witness.0f75b1.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem1, @ZZ.1(%Z.facet.f699e2.2) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.07c: <specific function> = specific_function %F, @F(%Z.facet.56d) [concrete]
// CHECK:STDOUT:   %F.specific_fn.b15: <specific function> = specific_function %F, @F(%Z.facet.501) [concrete]
// CHECK:STDOUT:   %F.specific_fn.07d: <specific function> = specific_function %F, @F(%Z.facet.b63) [concrete]
// CHECK:STDOUT:   %.a2c: type = fn_type_with_self_type %ZZ.type.df1, %Z.facet.56d [concrete]
// CHECK:STDOUT:   %.f35: type = fn_type_with_self_type %ZZ.type.df1, %Z.facet.501 [concrete]
// CHECK:STDOUT:   %.706: type = fn_type_with_self_type %ZZ.type.df1, %Z.facet.b63 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   impl_decl @impl.411 [concrete] {} {
// CHECK:STDOUT:     %.loc22_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_12.2: type = converted %.loc22_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc22_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_20.2: type = converted %.loc22_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc22_34: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.a62, element0 [symbolic_self = constants.%impl.elem0.7a8]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc22_28: type = where_expr %.Self [concrete = constants.%Z_where.type.7c5] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc22 = impl_witness_table (%impl_witness_assoc_constant.loc22, @impl.411.%ZZ.decl), @impl.411 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc22: <witness> = impl_witness %Z.impl_witness_table.loc22 [concrete = constants.%Z.impl_witness.bd6]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc22: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @impl.6ed [concrete] {} {
// CHECK:STDOUT:     %.loc26_13.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_13.2: type = converted %.loc26_13.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc26_22.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_22.2: type = converted %.loc26_22.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.aa.bb: type = struct_type {.aa: %empty_tuple.type, .bb: %empty_tuple.type} [concrete = constants.%struct_type.aa.bb]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc26_36: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.a62, element0 [symbolic_self = constants.%impl.elem0.7a8]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc26_30: type = where_expr %.Self [concrete = constants.%Z_where.type.47f] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %D.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc26 = impl_witness_table (%impl_witness_assoc_constant.loc26, @impl.6ed.%ZZ.decl), @impl.6ed [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc26: <witness> = impl_witness %Z.impl_witness_table.loc26 [concrete = constants.%Z.impl_witness.988]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc26: type = impl_witness_assoc_constant constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @impl.044 [concrete] {} {
// CHECK:STDOUT:     %.loc33_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc33_12.2: type = converted %.loc33_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc33_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc33_20.2: type = converted %.loc33_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %empty_tuple.type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self: %Z.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc33_34: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.a62, element0 [symbolic_self = constants.%impl.elem0.7a8]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc33_28: type = where_expr %.Self [concrete = constants.%Z_where.type.47f] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %D.ref
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc33 = impl_witness_table (%impl_witness_assoc_constant.loc33, @impl.044.%ZZ.decl), @impl.044 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc33: <witness> = impl_witness %Z.impl_witness_table.loc33 [concrete = constants.%Z.impl_witness.abe]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc33: type = impl_witness_assoc_constant constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.5af = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.c15a81.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.c15a81.2) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc37: %Z.type = name_ref T, %T.loc37_6.1 [symbolic = %T.loc37_6.2 (constants.%T)]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc37_17.1: type = facet_access_type %T.ref.loc37 [symbolic = %T.as_type.loc37_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc37: type = converted %T.ref.loc37, %T.as_type.loc37_17.1 [symbolic = %T.as_type.loc37_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc37_17.1: type = impl_witness_access constants.%Z.lookup_impl_witness.0f75b1.2, element0 [symbolic = %impl.elem0.loc37_17.2 (constants.%impl.elem0.4a5fb1.2)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %T.loc37_6.1: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc37_6.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %Z.assoc_type = assoc_entity element0, @Z.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ZZ.decl: %ZZ.type.df1 = fn_decl @ZZ.1 [concrete = constants.%ZZ.a9f] {
// CHECK:STDOUT:     %return.patt: @ZZ.1.%pattern_type (%pattern_type.c15a81.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @ZZ.1.%pattern_type (%pattern_type.c15a81.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc16_14.2: type = impl_witness_access constants.%Z.lookup_impl_witness.0f75b1.1, element0 [symbolic = %impl.elem0.loc16_14.1 (constants.%impl.elem0.4a5fb1.1)]
// CHECK:STDOUT:     %X.ref: type = name_ref X, %impl.elem0.loc16_14.2 [symbolic = %impl.elem0.loc16_14.1 (constants.%impl.elem0.4a5fb1.1)]
// CHECK:STDOUT:     %return.param: ref @ZZ.1.%impl.elem0.loc16_14.1 (%impl.elem0.4a5fb1.1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @ZZ.1.%impl.elem0.loc16_14.1 (%impl.elem0.4a5fb1.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %Z.assoc_type = assoc_entity element1, %ZZ.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   .ZZ = %assoc1
// CHECK:STDOUT:   witness = (%X, %ZZ.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.411: %struct_type.a.b as %.loc22_28 {
// CHECK:STDOUT:   %ZZ.decl: %ZZ.type.3f4 = fn_decl @ZZ.2 [concrete = constants.%ZZ.529] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc23_14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .ZZ = %ZZ.decl
// CHECK:STDOUT:   witness = file.%Z.impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6ed: %struct_type.aa.bb as %.loc26_30 {
// CHECK:STDOUT:   %ZZ.decl: %ZZ.type.4e2 = fn_decl @ZZ.3 [concrete = constants.%ZZ.8f1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc27_14: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .ZZ = %ZZ.decl
// CHECK:STDOUT:   witness = file.%Z.impl_witness.loc26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.044: %struct_type.b.a as %.loc33_28 {
// CHECK:STDOUT:   %ZZ.decl: %ZZ.type.559 = fn_decl @ZZ.4 [concrete = constants.%ZZ.0f1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc34_14: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .ZZ = %ZZ.decl
// CHECK:STDOUT:   witness = file.%Z.impl_witness.loc33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc19_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_19: type = converted %.loc19_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc19_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc20_18: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_19: type = converted %.loc20_18, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc20_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ZZ.1(@Z.%Self: %Z.type) {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.0f75b1.1)]
// CHECK:STDOUT:   %impl.elem0.loc16_14.1: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_14.1 (constants.%impl.elem0.4a5fb1.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc16_14.1 [symbolic = %pattern_type (constants.%pattern_type.c15a81.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @ZZ.1.%impl.elem0.loc16_14.1 (%impl.elem0.4a5fb1.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ZZ.2() -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %C.ref.loc23_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc23_28.1: %C = as_compatible %empty_tuple [concrete = constants.%empty_tuple.a69]
// CHECK:STDOUT:   %.loc23_28.2: %C = converted %.loc23_26, %.loc23_28.1 [concrete = constants.%empty_tuple.a69]
// CHECK:STDOUT:   return %.loc23_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ZZ.3() -> %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %D.ref.loc27_31: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc27_28.1: %D = as_compatible %empty_tuple [concrete = constants.%empty_tuple.bf9]
// CHECK:STDOUT:   %.loc27_28.2: %D = converted %.loc27_26, %.loc27_28.1 [concrete = constants.%empty_tuple.bf9]
// CHECK:STDOUT:   return %.loc27_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ZZ.4() -> %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %D.ref.loc34_31: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc34_28.1: %D = as_compatible %empty_tuple [concrete = constants.%empty_tuple.bf9]
// CHECK:STDOUT:   %.loc34_28.2: %D = converted %.loc34_26, %.loc34_28.1 [concrete = constants.%empty_tuple.bf9]
// CHECK:STDOUT:   return %.loc34_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc37_6.1: %Z.type) {
// CHECK:STDOUT:   %T.loc37_6.2: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc37_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc37_17.2: type = facet_access_type %T.loc37_6.2 [symbolic = %T.as_type.loc37_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc37_6.2, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.0f75b1.2)]
// CHECK:STDOUT:   %impl.elem0.loc37_17.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc37_17.2 (constants.%impl.elem0.4a5fb1.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc37_17.2 [symbolic = %pattern_type (constants.%pattern_type.c15a81.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %impl.elem0.loc37_17.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T.as_type.loc37_17.2, (%Z.lookup_impl_witness) [symbolic = %Z.facet (constants.%Z.facet.f699e2.2)]
// CHECK:STDOUT:   %.loc38_11.2: type = fn_type_with_self_type constants.%ZZ.type.df1, %Z.facet [symbolic = %.loc38_11.2 (constants.%.881)]
// CHECK:STDOUT:   %impl.elem1.loc38_11.2: @F.%.loc38_11.2 (%.881) = impl_witness_access %Z.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc38_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_11.2: <specific function> = specific_impl_function %impl.elem1.loc38_11.2, @ZZ.1(%Z.facet) [symbolic = %specific_impl_fn.loc38_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc38: %Z.type = name_ref T, %T.loc37_6.1 [symbolic = %T.loc37_6.2 (constants.%T)]
// CHECK:STDOUT:     %ZZ.ref: %Z.assoc_type = name_ref ZZ, @Z.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc38: type = facet_access_type %T.ref.loc38 [symbolic = %T.as_type.loc37_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc38_11.1: type = converted %T.ref.loc38, %T.as_type.loc38 [symbolic = %T.as_type.loc37_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc38_11.1: @F.%.loc38_11.2 (%.881) = impl_witness_access constants.%Z.lookup_impl_witness.0f75b1.2, element1 [symbolic = %impl.elem1.loc38_11.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_11.1: <specific function> = specific_impl_function %impl.elem1.loc38_11.1, @ZZ.1(constants.%Z.facet.f699e2.2) [symbolic = %specific_impl_fn.loc38_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc38_15: init @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) = call %specific_impl_fn.loc38_11.1()
// CHECK:STDOUT:     %.loc38_16.1: @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) = value_of_initializer %.loc38_15
// CHECK:STDOUT:     %.loc38_16.2: @F.%impl.elem0.loc37_17.2 (%impl.elem0.4a5fb1.2) = converted %.loc38_15, %.loc38_16.1
// CHECK:STDOUT:     return %.loc38_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.c48 = binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc42: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc42_23.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc42_23.2: type = converted %.loc42_23.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc42_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc42_31.2: type = converted %.loc42_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   %Z.facet.loc42: %Z.type = facet_value constants.%struct_type.a.b, (constants.%Z.impl_witness.bd6) [concrete = constants.%Z.facet.56d]
// CHECK:STDOUT:   %.loc42_33.1: %Z.type = converted %struct_type.a.b, %Z.facet.loc42 [concrete = constants.%Z.facet.56d]
// CHECK:STDOUT:   %F.specific_fn.loc42: <specific function> = specific_function %F.ref.loc42, @F(constants.%Z.facet.56d) [concrete = constants.%F.specific_fn.07c]
// CHECK:STDOUT:   %F.call.loc42: init %C = call %F.specific_fn.loc42()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc42_33.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc42_33.3: ref %C = temporary %.loc42_33.2, %F.call.loc42
// CHECK:STDOUT:   %c1: ref %C = bind_name c1, %.loc42_33.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d1.patt: %pattern_type.510 = binding_pattern d1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc43: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc43_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc43_24.2: type = converted %.loc43_24.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc43_33.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc43_33.2: type = converted %.loc43_33.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.aa.bb: type = struct_type {.aa: %empty_tuple.type, .bb: %empty_tuple.type} [concrete = constants.%struct_type.aa.bb]
// CHECK:STDOUT:   %Z.facet.loc43: %Z.type = facet_value constants.%struct_type.aa.bb, (constants.%Z.impl_witness.988) [concrete = constants.%Z.facet.501]
// CHECK:STDOUT:   %.loc43_35.1: %Z.type = converted %struct_type.aa.bb, %Z.facet.loc43 [concrete = constants.%Z.facet.501]
// CHECK:STDOUT:   %F.specific_fn.loc43: <specific function> = specific_function %F.ref.loc43, @F(constants.%Z.facet.501) [concrete = constants.%F.specific_fn.b15]
// CHECK:STDOUT:   %F.call.loc43: init %D = call %F.specific_fn.loc43()
// CHECK:STDOUT:   %D.ref.loc43: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %.loc43_35.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc43_35.3: ref %D = temporary %.loc43_35.2, %F.call.loc43
// CHECK:STDOUT:   %d1: ref %D = bind_name d1, %.loc43_35.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d2.patt: %pattern_type.510 = binding_pattern d2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc45: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc45_23.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc45_23.2: type = converted %.loc45_23.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc45_31.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc45_31.2: type = converted %.loc45_31.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: %empty_tuple.type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a]
// CHECK:STDOUT:   %Z.facet.loc45: %Z.type = facet_value constants.%struct_type.b.a, (constants.%Z.impl_witness.abe) [concrete = constants.%Z.facet.b63]
// CHECK:STDOUT:   %.loc45_33.1: %Z.type = converted %struct_type.b.a, %Z.facet.loc45 [concrete = constants.%Z.facet.b63]
// CHECK:STDOUT:   %F.specific_fn.loc45: <specific function> = specific_function %F.ref.loc45, @F(constants.%Z.facet.b63) [concrete = constants.%F.specific_fn.07d]
// CHECK:STDOUT:   %F.call.loc45: init %D = call %F.specific_fn.loc45()
// CHECK:STDOUT:   %D.ref.loc45: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %.loc45_33.2: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc45_33.3: ref %D = temporary %.loc45_33.2, %F.call.loc45
// CHECK:STDOUT:   %d2: ref %D = bind_name d2, %.loc45_33.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.f699e2.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ZZ.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.0f75b1.1
// CHECK:STDOUT:   %impl.elem0.loc16_14.1 => constants.%impl.elem0.4a5fb1.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c15a81.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.2a2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ZZ.1(constants.%Z.facet.56d) {
// CHECK:STDOUT:   %Self => constants.%Z.facet.56d
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.bd6
// CHECK:STDOUT:   %impl.elem0.loc16_14.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ZZ.1(constants.%Z.facet.501) {
// CHECK:STDOUT:   %Self => constants.%Z.facet.501
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.988
// CHECK:STDOUT:   %impl.elem0.loc16_14.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ZZ.1(constants.%Z.facet.b63) {
// CHECK:STDOUT:   %Self => constants.%Z.facet.b63
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.abe
// CHECK:STDOUT:   %impl.elem0.loc16_14.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Z.facet.f699e2.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc37_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc37_17.2 => constants.%T.as_type
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.0f75b1.2
// CHECK:STDOUT:   %impl.elem0.loc37_17.2 => constants.%impl.elem0.4a5fb1.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c15a81.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ZZ.1(constants.%Z.facet.f699e2.2) {
// CHECK:STDOUT:   %Self => constants.%Z.facet.f699e2.2
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.0f75b1.2
// CHECK:STDOUT:   %impl.elem0.loc16_14.1 => constants.%impl.elem0.4a5fb1.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c15a81.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Z.facet.56d) {
// CHECK:STDOUT:   %T.loc37_6.2 => constants.%Z.facet.56d
// CHECK:STDOUT:   %T.as_type.loc37_17.2 => constants.%struct_type.a.b
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.bd6
// CHECK:STDOUT:   %impl.elem0.loc37_17.2 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.56d
// CHECK:STDOUT:   %.loc38_11.2 => constants.%.a2c
// CHECK:STDOUT:   %impl.elem1.loc38_11.2 => constants.%ZZ.529
// CHECK:STDOUT:   %specific_impl_fn.loc38_11.2 => constants.%ZZ.529
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Z.facet.501) {
// CHECK:STDOUT:   %T.loc37_6.2 => constants.%Z.facet.501
// CHECK:STDOUT:   %T.as_type.loc37_17.2 => constants.%struct_type.aa.bb
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.988
// CHECK:STDOUT:   %impl.elem0.loc37_17.2 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.501
// CHECK:STDOUT:   %.loc38_11.2 => constants.%.f35
// CHECK:STDOUT:   %impl.elem1.loc38_11.2 => constants.%ZZ.8f1
// CHECK:STDOUT:   %specific_impl_fn.loc38_11.2 => constants.%ZZ.8f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Z.facet.b63) {
// CHECK:STDOUT:   %T.loc37_6.2 => constants.%Z.facet.b63
// CHECK:STDOUT:   %T.as_type.loc37_17.2 => constants.%struct_type.b.a
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.impl_witness.abe
// CHECK:STDOUT:   %impl.elem0.loc37_17.2 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.510
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.b63
// CHECK:STDOUT:   %.loc38_11.2 => constants.%.706
// CHECK:STDOUT:   %impl.elem1.loc38_11.2 => constants.%ZZ.0f1
// CHECK:STDOUT:   %specific_impl_fn.loc38_11.2 => constants.%ZZ.0f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %Self.as_type.7f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type.419 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Self.as_type.560 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.220: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%BitAnd.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness %BitAnd.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest.loc9_14.2) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Op.1.%pattern_type (%pattern_type.a80) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.220]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T.loc21_14.2) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc21_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%BitAnd.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:   %pattern_type.loc10_14: type = pattern_type %Self.as_type.loc10_20.1 [symbolic = %pattern_type.loc10_14 (constants.%pattern_type.947)]
// CHECK:STDOUT:   %pattern_type.loc10_28: type = pattern_type %Dest [symbolic = %pattern_type.loc10_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)) -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:   %pattern_type.loc14_14: type = pattern_type %Self.as_type.loc14_20.1 [symbolic = %pattern_type.loc14_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc14_28: type = pattern_type %Dest [symbolic = %pattern_type.loc14_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)) -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc18_15.1 [symbolic = %pattern_type (constants.%pattern_type.a80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560), %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T), %other.param: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT:   %pattern_type.loc10_14 => constants.%pattern_type.947
// CHECK:STDOUT:   %pattern_type.loc10_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc14_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

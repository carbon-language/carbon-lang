// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon

// --- name_lookup_success.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  T.F();
  T.(I.F)();

  T.(J.G)();
}

// --- fail_name_lookup.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup.carbon:[[@LINE+4]]:3: error: member name `G` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- name_lookup_instance_success.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  x.FNonInstance();
  x.FSelf();
  x.(I.FSelf)();

  x.(J.GSelf)();
}

// --- fail_name_lookup_instance.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GNonInstance` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GNonInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  x.GNonInstance();

  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GSelf` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GSelf();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.GSelf();
}

// CHECK:STDOUT: --- name_lookup_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.a5e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.4b0: %J.assoc_type = assoc_entity element0, @J.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.5a0: type = fn_type_with_self_type %F.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.5a0 = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.577: <specific function> = specific_impl_function %impl.elem0.594, @F(%I.facet) [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface1: <witness> = facet_access_witness %T, element1 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type, (%T.as_wit.iface1) [symbolic]
// CHECK:STDOUT:   %.201: type = fn_type_with_self_type %G.type, %J.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.ff4: %.201 = impl_witness_access %T.as_wit.iface1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.907: <specific function> = specific_impl_function %impl.elem0.ff4, @G(%J.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc11: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc11_22, %J.ref.loc11
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc11_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc12_4.2: type = facet_access_type %T.loc11_10.2 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc12_4.2: <witness> = facet_access_witness %T.loc11_10.2, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %I.facet.loc12: %I.type.733 = facet_value %T.as_type.loc12_4.2, (%T.as_wit.iface0.loc12_4.2) [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:   %.loc12_4.2: type = fn_type_with_self_type constants.%F.type, %I.facet.loc12 [symbolic = %.loc12_4.2 (constants.%.5a0)]
// CHECK:STDOUT:   %impl.elem0.loc12_4.2: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %T.as_wit.iface0.loc12_4.2, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_4.2: <specific function> = specific_impl_function %impl.elem0.loc12_4.2, @F(%I.facet.loc12) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:   %T.as_wit.iface1.loc15_4.2: <witness> = facet_access_witness %T.loc11_10.2, element1 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:   %J.facet.loc15_4.2: %J.type = facet_value %T.as_type.loc12_4.2, (%T.as_wit.iface1.loc15_4.2) [symbolic = %J.facet.loc15_4.2 (constants.%J.facet)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type constants.%G.type, %J.facet.loc15_4.2 [symbolic = %.loc15_4.2 (constants.%.201)]
// CHECK:STDOUT:   %impl.elem0.loc15_4.2: @INotJ.%.loc15_4.2 (%.201) = impl_witness_access %T.as_wit.iface1.loc15_4.2, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0.ff4)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem0.loc15_4.2, @G(%J.facet.loc15_4.2) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.907)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc12: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref.loc12: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:     %T.as_type.loc12_4.1: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_4.1: type = converted %T.ref.loc12, %T.as_type.loc12_4.1 [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc12_4.1: <witness> = facet_access_witness %T.ref.loc12, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc12_4.1: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %T.as_wit.iface0.loc12_4.1, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %specific_impl_fn.loc12_4.1: <specific function> = specific_impl_function %impl.elem0.loc12_4.1, @F(constants.%I.facet) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:     %.loc12_7: init %empty_tuple.type = call %specific_impl_fn.loc12_4.1()
// CHECK:STDOUT:     %T.ref.loc13: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:     %F.ref.loc13: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:     %T.as_wit.iface0.loc13: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type constants.%T [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.facet.loc13: %I.type.733 = facet_value %T.as_type.loc13, (%T.as_wit.iface0.loc13) [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:     %.loc13_4: %I.type.733 = converted %T.ref.loc13, %I.facet.loc13 [symbolic = %I.facet.loc12 (constants.%I.facet)]
// CHECK:STDOUT:     %as_wit.iface0.loc13: <witness> = facet_access_witness %.loc13_4, element0 [symbolic = %T.as_wit.iface0.loc12_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc13: @INotJ.%.loc12_4.2 (%.5a0) = impl_witness_access %as_wit.iface0.loc13, element0 [symbolic = %impl.elem0.loc12_4.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %specific_impl_fn.loc13: <specific function> = specific_impl_function %impl.elem0.loc13, @F(constants.%I.facet) [symbolic = %specific_impl_fn.loc12_4.2 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:     %.loc13_11: init %empty_tuple.type = call %specific_impl_fn.loc13()
// CHECK:STDOUT:     %T.ref.loc15: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:     %J.ref.loc15: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %G.ref: %J.assoc_type = name_ref G, @J.%assoc0 [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:     %T.as_wit.iface1.loc15_4.1: <witness> = facet_access_witness constants.%T, element1 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type constants.%T [symbolic = %T.as_type.loc12_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.facet.loc15_4.1: %J.type = facet_value %T.as_type.loc15, (%T.as_wit.iface1.loc15_4.1) [symbolic = %J.facet.loc15_4.2 (constants.%J.facet)]
// CHECK:STDOUT:     %.loc15_4.1: %J.type = converted %T.ref.loc15, %J.facet.loc15_4.1 [symbolic = %J.facet.loc15_4.2 (constants.%J.facet)]
// CHECK:STDOUT:     %as_wit.iface0.loc15: <witness> = facet_access_witness %.loc15_4.1, element0 [symbolic = %T.as_wit.iface1.loc15_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem0.loc15_4.1: @INotJ.%.loc15_4.2 (%.201) = impl_witness_access %as_wit.iface0.loc15, element0 [symbolic = %impl.elem0.loc15_4.2 (constants.%impl.elem0.ff4)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem0.loc15_4.1, @G(constants.%J.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.907)]
// CHECK:STDOUT:     %.loc15_11: init %empty_tuple.type = call %specific_impl_fn.loc15_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc11_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@INotJ.%I.facet.loc12) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(@INotJ.%J.facet.loc15_4.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.a5e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.4b0: %J.assoc_type = assoc_entity element0, @J.%G.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc11_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: type = splice_block %.loc11_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc11_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc11_22, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.a5e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = <poisoned>
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %G.decl [concrete = constants.%assoc0.4b0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc11_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_10.1: %I.type.100) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type.100 = name_ref T, %T.loc11_10.1 [symbolic = %T.loc11_10.2 (constants.%T)]
// CHECK:STDOUT:     %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc11_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_10.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- name_lookup_instance_success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %FNonInstance.type: type = fn_type @FNonInstance [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FNonInstance: %FNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.8b6: %I.assoc_type = assoc_entity element0, @I.%FNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %FSelf.type: type = fn_type @FSelf [concrete]
// CHECK:STDOUT:   %FSelf: %FSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.685: %I.assoc_type = assoc_entity element1, @I.%FSelf.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %GNonInstance.type: type = fn_type @GNonInstance [concrete]
// CHECK:STDOUT:   %GNonInstance: %GNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.a91: %J.assoc_type = assoc_entity element0, @J.%GNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %GSelf.type: type = fn_type @GSelf [concrete]
// CHECK:STDOUT:   %GSelf: %GSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.26a: %J.assoc_type = assoc_entity element1, @J.%GSelf.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T, element0 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %.b38: type = fn_type_with_self_type %FNonInstance.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b38 = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e7d: <specific function> = specific_impl_function %impl.elem0, @FNonInstance(%I.facet) [symbolic]
// CHECK:STDOUT:   %.e5a: type = fn_type_with_self_type %FSelf.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.d3e: %.e5a = impl_witness_access %T.as_wit.iface0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3d3: <specific function> = specific_impl_function %impl.elem1.d3e, @FSelf(%I.facet) [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface1: <witness> = facet_access_witness %T, element1 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type, (%T.as_wit.iface1) [symbolic]
// CHECK:STDOUT:   %.5b4: type = fn_type_with_self_type %GSelf.type, %J.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1.d26: %.5b4 = impl_witness_access %T.as_wit.iface1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.83c: <specific function> = specific_impl_function %impl.elem1.d26, @GSelf(%J.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc13_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_16.1: type = splice_block %.loc13_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref.loc13: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc13_22, %J.ref.loc13
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_40.1: type = splice_block %.loc13_40.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.100 = name_ref T, %T.loc13_10.1 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc13_40.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_40.2: type = converted %T.ref, %T.as_type.loc13_40.1 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %FNonInstance.decl: %FNonInstance.type = fn_decl @FNonInstance [concrete = constants.%FNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %FNonInstance.decl [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:   %FSelf.decl: %FSelf.type = fn_decl @FSelf [concrete = constants.%FSelf] {
// CHECK:STDOUT:     %self.patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type.733 = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_18.2: type = converted %Self.ref, %Self.as_type.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %FSelf.decl [concrete = constants.%assoc1.685]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .FNonInstance = %assoc0
// CHECK:STDOUT:   .FSelf = %assoc1
// CHECK:STDOUT:   witness = (%FNonInstance.decl, %FSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %GNonInstance.decl: %GNonInstance.type = fn_decl @GNonInstance [concrete = constants.%GNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %GNonInstance.decl [concrete = constants.%assoc0.a91]
// CHECK:STDOUT:   %GSelf.decl: %GSelf.type = fn_decl @GSelf [concrete = constants.%GSelf] {
// CHECK:STDOUT:     %self.patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc10_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc10_18.2: type = converted %Self.ref, %Self.as_type.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %GSelf.decl [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .GNonInstance = %assoc0
// CHECK:STDOUT:   .GSelf = %assoc1
// CHECK:STDOUT:   witness = (%GNonInstance.decl, %GSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FNonInstance(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FSelf(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GNonInstance(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GSelf(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc13_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc13_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc13_40.2: type = facet_access_type %T.loc13_10.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @INotJ.%T.as_type.loc13_40.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc14_4.2: <witness> = facet_access_witness %T.loc13_10.2, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %I.facet: %I.type.733 = facet_value %T.as_type.loc13_40.2, (%T.as_wit.iface0.loc14_4.2) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc14_4.2: type = fn_type_with_self_type constants.%FNonInstance.type, %I.facet [symbolic = %.loc14_4.2 (constants.%.b38)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @INotJ.%.loc14_4.2 (%.b38) = impl_witness_access %T.as_wit.iface0.loc14_4.2, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @FNonInstance(%I.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e7d)]
// CHECK:STDOUT:   %.loc15_4.2: type = fn_type_with_self_type constants.%FSelf.type, %I.facet [symbolic = %.loc15_4.2 (constants.%.e5a)]
// CHECK:STDOUT:   %impl.elem1.loc15_4.2: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc14_4.2, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_4.2: <specific function> = specific_impl_function %impl.elem1.loc15_4.2, @FSelf(%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:   %T.as_wit.iface1.loc18_4.2: <witness> = facet_access_witness %T.loc13_10.2, element1 [symbolic = %T.as_wit.iface1.loc18_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.as_type.loc13_40.2, (%T.as_wit.iface1.loc18_4.2) [symbolic = %J.facet (constants.%J.facet)]
// CHECK:STDOUT:   %.loc18_4: type = fn_type_with_self_type constants.%GSelf.type, %J.facet [symbolic = %.loc18_4 (constants.%.5b4)]
// CHECK:STDOUT:   %impl.elem1.loc18_4.2: @INotJ.%.loc18_4 (%.5b4) = impl_witness_access %T.as_wit.iface1.loc18_4.2, element1 [symbolic = %impl.elem1.loc18_4.2 (constants.%impl.elem1.d26)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_4.2: <specific function> = specific_impl_function %impl.elem1.loc18_4.2, @GSelf(%J.facet) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.83c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_10.1: %I.type.100](%x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc14: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %FNonInstance.ref: %I.assoc_type = name_ref FNonInstance, @I.%assoc0 [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type constants.%T [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc14_4.1: type = converted constants.%T, %T.as_type.loc14 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc14_4.1: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @INotJ.%.loc14_4.2 (%.b38) = impl_witness_access %T.as_wit.iface0.loc14_4.1, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @FNonInstance(constants.%I.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.e7d)]
// CHECK:STDOUT:     %.loc14_18: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     %x.ref.loc15: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %FSelf.ref.loc15: %I.assoc_type = name_ref FSelf, @I.%assoc1 [concrete = constants.%assoc1.685]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type constants.%T [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15_4.1: type = converted constants.%T, %T.as_type.loc15 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.iface0.loc15: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc15_4.1: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc15, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:     %bound_method.loc15_4: <bound method> = bound_method %x.ref.loc15, %impl.elem1.loc15_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_4.1: <specific function> = specific_impl_function %impl.elem1.loc15_4.1, @FSelf(constants.%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:     %bound_method.loc15_11: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_4.1
// CHECK:STDOUT:     %.loc15_11: init %empty_tuple.type = call %bound_method.loc15_11(%x.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %I.ref.loc16: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:     %FSelf.ref.loc16: %I.assoc_type = name_ref FSelf, @I.%assoc1 [concrete = constants.%assoc1.685]
// CHECK:STDOUT:     %T.as_wit.iface0.loc16: <witness> = facet_access_witness constants.%T, element0 [symbolic = %T.as_wit.iface0.loc14_4.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem1.loc16: @INotJ.%.loc15_4.2 (%.e5a) = impl_witness_access %T.as_wit.iface0.loc16, element1 [symbolic = %impl.elem1.loc15_4.2 (constants.%impl.elem1.d3e)]
// CHECK:STDOUT:     %bound_method.loc16_4: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16
// CHECK:STDOUT:     %specific_impl_fn.loc16: <specific function> = specific_impl_function %impl.elem1.loc16, @FSelf(constants.%I.facet) [symbolic = %specific_impl_fn.loc15_4.2 (constants.%specific_impl_fn.3d3)]
// CHECK:STDOUT:     %bound_method.loc16_15: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16
// CHECK:STDOUT:     %.loc16: init %empty_tuple.type = call %bound_method.loc16_15(%x.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc18: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %J.ref.loc18: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %GSelf.ref: %J.assoc_type = name_ref GSelf, @J.%assoc1 [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:     %T.as_wit.iface1.loc18_4.1: <witness> = facet_access_witness constants.%T, element1 [symbolic = %T.as_wit.iface1.loc18_4.2 (constants.%T.as_wit.iface1)]
// CHECK:STDOUT:     %impl.elem1.loc18_4.1: @INotJ.%.loc18_4 (%.5b4) = impl_witness_access %T.as_wit.iface1.loc18_4.1, element1 [symbolic = %impl.elem1.loc18_4.2 (constants.%impl.elem1.d26)]
// CHECK:STDOUT:     %bound_method.loc18_4: <bound method> = bound_method %x.ref.loc18, %impl.elem1.loc18_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_4.1: <specific function> = specific_impl_function %impl.elem1.loc18_4.1, @GSelf(constants.%J.facet) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.83c)]
// CHECK:STDOUT:     %bound_method.loc18_15: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_4.1
// CHECK:STDOUT:     %.loc18_15: init %empty_tuple.type = call %bound_method.loc18_15(%x.ref.loc18)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%Self.as_type.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GNonInstance(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%Self.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc13_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc13_40.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%J.facet) {
// CHECK:STDOUT:   %Self => constants.%J.facet
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(@INotJ.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(@INotJ.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(@INotJ.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_name_lookup_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %FNonInstance.type: type = fn_type @FNonInstance [concrete]
// CHECK:STDOUT:   %FNonInstance: %FNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.733 [concrete]
// CHECK:STDOUT:   %assoc0.8b6: %I.assoc_type = assoc_entity element0, @I.%FNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %FSelf.type: type = fn_type @FSelf [concrete]
// CHECK:STDOUT:   %FSelf: %FSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.685: %I.assoc_type = assoc_entity element1, @I.%FSelf.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %GNonInstance.type: type = fn_type @GNonInstance [concrete]
// CHECK:STDOUT:   %GNonInstance: %GNonInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.a91: %J.assoc_type = assoc_entity element0, @J.%GNonInstance.decl [concrete]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %GSelf.type: type = fn_type @GSelf [concrete]
// CHECK:STDOUT:   %GSelf: %GSelf.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1.26a: %J.assoc_type = assoc_entity element1, @J.%GSelf.decl [concrete]
// CHECK:STDOUT:   %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.type.100: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I.type.100 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .INotJ = %INotJ.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt.loc13_10.1: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_16.1: type = splice_block %.loc13_16.2 [concrete = constants.%I.type.100] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %.Self: %I.type.733 = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref: %I.type.733 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc13_16.2: type = where_expr %.Self [concrete = constants.%I.type.100] {
// CHECK:STDOUT:         requirement_impls %.loc13_22, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_10.1: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_40.1: type = splice_block %.loc13_40.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type.100 = name_ref T, %T.loc13_10.1 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc13_40.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_40.2: type = converted %T.ref, %T.as_type.loc13_40.1 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %FNonInstance.decl: %FNonInstance.type = fn_decl @FNonInstance [concrete = constants.%FNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %FNonInstance.decl [concrete = constants.%assoc0.8b6]
// CHECK:STDOUT:   %FSelf.decl: %FSelf.type = fn_decl @FSelf [concrete = constants.%FSelf] {
// CHECK:STDOUT:     %self.patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type.733 = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_18.2: type = converted %Self.ref, %Self.as_type.loc5_18.2 [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %FSelf.decl [concrete = constants.%assoc1.685]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .FNonInstance = %assoc0
// CHECK:STDOUT:   .FSelf = %assoc1
// CHECK:STDOUT:   .GNonInstance = <poisoned>
// CHECK:STDOUT:   .GSelf = <poisoned>
// CHECK:STDOUT:   witness = (%FNonInstance.decl, %FSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %GNonInstance.decl: %GNonInstance.type = fn_decl @GNonInstance [concrete = constants.%GNonInstance] {} {}
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %GNonInstance.decl [concrete = constants.%assoc0.a91]
// CHECK:STDOUT:   %GSelf.decl: %GSelf.type = fn_decl @GSelf [concrete = constants.%GSelf] {
// CHECK:STDOUT:     %self.patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc10_18.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc10_18.2: type = converted %Self.ref, %Self.as_type.loc10_18.2 [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %GSelf.decl [concrete = constants.%assoc1.26a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .GNonInstance = %assoc0
// CHECK:STDOUT:   .GSelf = %assoc1
// CHECK:STDOUT:   witness = (%GNonInstance.decl, %GSelf.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FNonInstance(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FSelf(@I.%Self: %I.type.733) {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @FSelf.%Self.as_type.loc5_18.1 (%Self.as_type.b70)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GNonInstance(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GSelf(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GSelf.%Self.as_type.loc10_18.1 (%Self.as_type.3df)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc13_10.1: %I.type.100) {
// CHECK:STDOUT:   %T.loc13_10.2: %I.type.100 = bind_symbolic_name T, 0 [symbolic = %T.loc13_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_10.2: %I.type.100 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc13_40.2: type = facet_access_type %T.loc13_10.2 [symbolic = %T.as_type.loc13_40.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @INotJ.%T.as_type.loc13_40.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_10.1: %I.type.100](%x.param_patt: @INotJ.%T.as_type.loc13_40.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc18: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %GNonInstance.ref: <error> = name_ref GNonInstance, <error> [concrete = <error>]
// CHECK:STDOUT:     %x.ref.loc24: @INotJ.%T.as_type.loc13_40.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %GSelf.ref: <error> = name_ref GSelf, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FNonInstance(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FSelf(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_18.1 => constants.%Self.as_type.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GNonInstance(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GSelf(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc10_18.1 => constants.%Self.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc13_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc13_40.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

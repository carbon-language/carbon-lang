// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/dot_self_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/dot_self_index.carbon

interface Empty(W:! type) {
  let A:! type;
}

// T has index 0
// .Self has index invalid, not 1
// V has index 1, does not match .Self
fn H(T:! type, U: Empty(T) where .A = T*, V:! type) {}

fn G(U: Empty(i32) where .A = i32*) {
  H(i32, U, bool);
}

// CHECK:STDOUT: --- dot_self_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.d5a: type = generic_interface_type @Empty [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Empty.generic: %Empty.type.d5a = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type.3e5fde.1: type = facet_type <@Empty, @Empty(%W)> [symbolic]
// CHECK:STDOUT:   %Self.8c5170.1: %Empty.type.3e5fde.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.54c14f.1: type = assoc_entity_type %Empty.type.3e5fde.1 [symbolic]
// CHECK:STDOUT:   %assoc0.68f673.1: %Empty.assoc_type.54c14f.1 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.3e5fde.2: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.c95: %Empty.type.3e5fde.2 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.8c5170.2: %Empty.type.3e5fde.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.54c14f.2: type = assoc_entity_type %Empty.type.3e5fde.2 [symbolic]
// CHECK:STDOUT:   %assoc0.68f673.2: %Empty.assoc_type.54c14f.2 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %require_complete.22f: <witness> = require_complete_type %Empty.type.3e5fde.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.a75: type = facet_access_type %.Self.c95 [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.iface0.663: <witness> = facet_access_witness %.Self.c95, element0 [symbolic]
// CHECK:STDOUT:   %Empty.facet.782: %Empty.type.3e5fde.2 = facet_value %.Self.as_type.a75, (%.Self.as_wit.iface0.663) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d0b: type = impl_witness_access %.Self.as_wit.iface0.663, element0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Empty_where.type.5a8: type = facet_type <@Empty, @Empty(%T) where %impl.elem0.d0b = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.796: <witness> = require_complete_type %Empty_where.type.5a8 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Empty.type.f0b: type = facet_type <@Empty, @Empty(%i32)> [concrete]
// CHECK:STDOUT:   %.Self.e6e: %Empty.type.f0b = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.dcf: %Empty.type.f0b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.a46: type = assoc_entity_type %Empty.type.f0b [concrete]
// CHECK:STDOUT:   %assoc0.0c3: %Empty.assoc_type.a46 = assoc_entity element0, @Empty.%A [concrete]
// CHECK:STDOUT:   %.Self.as_type.920: type = facet_access_type %.Self.e6e [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0.67d: <witness> = facet_access_witness %.Self.e6e, element0 [symbolic_self]
// CHECK:STDOUT:   %Empty.facet.311: %Empty.type.f0b = facet_value %.Self.as_type.920, (%.Self.as_wit.iface0.67d) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1bd: type = impl_witness_access %.Self.as_wit.iface0.67d, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Empty_where.type.52c: type = facet_type <@Empty, @Empty(%i32) where %impl.elem0.1bd = %ptr.235> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.091: <witness> = complete_type_witness %Empty.type.f0b [concrete]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H, @H(%i32, bool) [concrete]
// CHECK:STDOUT:   %complete_type.6bd: <witness> = complete_type_witness %Empty_where.type.52c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: %Empty.type.d5a = interface_decl @Empty [concrete = constants.%Empty.generic] {
// CHECK:STDOUT:     %W.patt.loc11_17.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.loc11_17.1: type = bind_symbolic_name W, 0 [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt.loc18_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt: @H.%Empty_where.type (%Empty_where.type.5a8) = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: @H.%Empty_where.type (%Empty_where.type.5a8) = value_param_pattern %U.patt, call_param0
// CHECK:STDOUT:     %V.patt.loc18_43.1: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc18_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.5a8) = value_param call_param0
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Empty_where.type (constants.%Empty_where.type.5a8)] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %T.ref.loc18_25: type = name_ref T, %T.loc18_6.1 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:       %Empty.type.loc18_26.1: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc18_26.2 (constants.%Empty.type.3e5fde.2)]
// CHECK:STDOUT:       %.Self.1: @H.%Empty.type.loc18_26.2 (%Empty.type.3e5fde.2) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.c95)]
// CHECK:STDOUT:       %.Self.ref: @H.%Empty.type.loc18_26.2 (%Empty.type.3e5fde.2) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.c95)]
// CHECK:STDOUT:       %.loc18_34.1: @H.%Empty.assoc_type (%Empty.assoc_type.54c14f.2) = specific_constant @A.%assoc0, @Empty(constants.%T) [symbolic = %assoc0 (constants.%assoc0.68f673.2)]
// CHECK:STDOUT:       %A.ref: @H.%Empty.assoc_type (%Empty.assoc_type.54c14f.2) = name_ref A, %.loc18_34.1 [symbolic = %assoc0 (constants.%assoc0.68f673.2)]
// CHECK:STDOUT:       %.Self.as_type.loc18_34.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc18_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:       %.loc18_34.2: type = converted %.Self.ref, %.Self.as_type.loc18_34.1 [symbolic = %.Self.as_type.loc18_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:       %.Self.as_wit.iface0.loc18_34.1: <witness> = facet_access_witness %.Self.ref, element0 [symbolic = %.Self.as_wit.iface0.loc18_34.2 (constants.%.Self.as_wit.iface0.663)]
// CHECK:STDOUT:       %impl.elem0.loc18_34.1: type = impl_witness_access %.Self.as_wit.iface0.loc18_34.1, element0 [symbolic = %impl.elem0.loc18_34.2 (constants.%impl.elem0.d0b)]
// CHECK:STDOUT:       %T.ref.loc18_39: type = name_ref T, %T.loc18_6.1 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc18_40.1: type = ptr_type %T [symbolic = %ptr.loc18_40.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %.loc18_28.2: type = where_expr %.Self.1 [symbolic = %Empty_where.type (constants.%Empty_where.type.5a8)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_34.1, %ptr.loc18_40.1
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.5a8) = bind_name U, %U.param
// CHECK:STDOUT:     %V.loc18_43.1: type = bind_symbolic_name V, 1 [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: %Empty_where.type.52c = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: %Empty_where.type.52c = value_param_pattern %U.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: %Empty_where.type.52c = value_param call_param0
// CHECK:STDOUT:     %.loc20_20.1: type = splice_block %.loc20_20.2 [concrete = constants.%Empty_where.type.52c] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %int_32.loc20_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Empty.type: type = facet_type <@Empty, @Empty(constants.%i32)> [concrete = constants.%Empty.type.f0b]
// CHECK:STDOUT:       %.Self: %Empty.type.f0b = bind_symbolic_name .Self [symbolic_self = constants.%.Self.e6e]
// CHECK:STDOUT:       %.Self.ref: %Empty.type.f0b = name_ref .Self, %.Self [symbolic_self = constants.%.Self.e6e]
// CHECK:STDOUT:       %.loc20_26.1: %Empty.assoc_type.a46 = specific_constant @A.%assoc0, @Empty(constants.%i32) [concrete = constants.%assoc0.0c3]
// CHECK:STDOUT:       %A.ref: %Empty.assoc_type.a46 = name_ref A, %.loc20_26.1 [concrete = constants.%assoc0.0c3]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.920]
// CHECK:STDOUT:       %.loc20_26.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.920]
// CHECK:STDOUT:       %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0.67d]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.1bd]
// CHECK:STDOUT:       %int_32.loc20_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:       %.loc20_20.2: type = where_expr %.Self [concrete = constants.%Empty_where.type.52c] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %ptr
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: %Empty_where.type.52c = bind_name U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Empty(%W.loc11_17.1: type) {
// CHECK:STDOUT:   %W.loc11_17.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_17.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty, @Empty(%W.loc11_17.2)> [symbolic = %Empty.type (constants.%Empty.type.3e5fde.1)]
// CHECK:STDOUT:   %Self.2: @Empty.%Empty.type (%Empty.type.3e5fde.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8c5170.1)]
// CHECK:STDOUT:   %Empty.assoc_type: type = assoc_entity_type @Empty.%Empty.type (%Empty.type.3e5fde.1) [symbolic = %Empty.assoc_type (constants.%Empty.assoc_type.54c14f.1)]
// CHECK:STDOUT:   %assoc0: @Empty.%Empty.assoc_type (%Empty.assoc_type.54c14f.1) = assoc_entity element0, %A [symbolic = %assoc0 (constants.%assoc0.68f673.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Empty.%Empty.type (%Empty.type.3e5fde.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8c5170.1)]
// CHECK:STDOUT:     %A: type = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:       %assoc0: @Empty.%Empty.assoc_type (%Empty.assoc_type.54c14f.1) = assoc_entity element0, @Empty.%A [symbolic = @Empty.%assoc0 (constants.%assoc0.68f673.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .A = @A.%assoc0
// CHECK:STDOUT:     witness = (%A)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @A(@Empty.%W.loc11_17.1: type, @Empty.%Self.1: @Empty.%Empty.type (%Empty.type.3e5fde.1)) {
// CHECK:STDOUT:   assoc_const A:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc18_6.1: type, %V.loc18_43.1: type) {
// CHECK:STDOUT:   %T.loc18_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc18_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Empty.type.loc18_26.2: type = facet_type <@Empty, @Empty(%T.loc18_6.2)> [symbolic = %Empty.type.loc18_26.2 (constants.%Empty.type.3e5fde.2)]
// CHECK:STDOUT:   %.Self.2: @H.%Empty.type.loc18_26.2 (%Empty.type.3e5fde.2) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.c95)]
// CHECK:STDOUT:   %require_complete.loc18_34: <witness> = require_complete_type @H.%Empty.type.loc18_26.2 (%Empty.type.3e5fde.2) [symbolic = %require_complete.loc18_34 (constants.%require_complete.22f)]
// CHECK:STDOUT:   %Empty.assoc_type: type = assoc_entity_type @H.%Empty.type.loc18_26.2 (%Empty.type.3e5fde.2) [symbolic = %Empty.assoc_type (constants.%Empty.assoc_type.54c14f.2)]
// CHECK:STDOUT:   %assoc0: @H.%Empty.assoc_type (%Empty.assoc_type.54c14f.2) = assoc_entity element0, @Empty.%A [symbolic = %assoc0 (constants.%assoc0.68f673.2)]
// CHECK:STDOUT:   %.Self.as_type.loc18_34.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc18_34.2 (constants.%.Self.as_type.a75)]
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc18_34.2: <witness> = facet_access_witness %.Self.2, element0 [symbolic = %.Self.as_wit.iface0.loc18_34.2 (constants.%.Self.as_wit.iface0.663)]
// CHECK:STDOUT:   %impl.elem0.loc18_34.2: type = impl_witness_access %.Self.as_wit.iface0.loc18_34.2, element0 [symbolic = %impl.elem0.loc18_34.2 (constants.%impl.elem0.d0b)]
// CHECK:STDOUT:   %ptr.loc18_40.2: type = ptr_type @H.%T.loc18_6.2 (%T) [symbolic = %ptr.loc18_40.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc18_6.2) where %impl.elem0.loc18_34.2 (constants.%impl.elem0.d0b) = %ptr.loc18_40.2 (constants.%ptr.79f)> [symbolic = %Empty_where.type (constants.%Empty_where.type.5a8)]
// CHECK:STDOUT:   %V.loc18_43.2: type = bind_symbolic_name V, 1 [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc18_43.2: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_17: <witness> = require_complete_type @H.%Empty_where.type (%Empty_where.type.5a8) [symbolic = %require_complete.loc18_17 (constants.%require_complete.796)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc18_6.1: type, %U.param_patt: @H.%Empty_where.type (%Empty_where.type.5a8), %V.patt.loc18_43.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%U.param_patt: %Empty_where.type.52c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %U.ref: %Empty_where.type.52c = name_ref U, %U
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc21_17.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:   %.loc21_17.2: type = converted %bool.make_type, %.loc21_17.1 [concrete = bool]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H.ref, @H(constants.%i32, bool) [concrete = constants.%H.specific_fn]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %H.specific_fn(%U.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%W) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W, constants.%Self.8c5170.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(%W.loc11_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.3e5fde.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.8c5170.2
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.54c14f.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.68f673.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T, constants.%Empty.facet.782) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%T.patt
// CHECK:STDOUT:   %Empty.type.loc18_26.2 => constants.%Empty.type.3e5fde.2
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.c95
// CHECK:STDOUT:   %require_complete.loc18_34 => constants.%require_complete.22f
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.54c14f.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.68f673.2
// CHECK:STDOUT:   %.Self.as_type.loc18_34.2 => constants.%.Self.as_type.a75
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc18_34.2 => constants.%.Self.as_wit.iface0.663
// CHECK:STDOUT:   %impl.elem0.loc18_34.2 => constants.%impl.elem0.d0b
// CHECK:STDOUT:   %ptr.loc18_40.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.5a8
// CHECK:STDOUT:   %V.loc18_43.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc18_43.2 => constants.%V.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(@H.%T.loc18_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%i32) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%i32
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.f0b
// CHECK:STDOUT:   %Self.2 => constants.%Self.dcf
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.a46
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%i32, constants.%Empty.facet.311) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32, bool) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%T.patt
// CHECK:STDOUT:   %Empty.type.loc18_26.2 => constants.%Empty.type.f0b
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.e6e
// CHECK:STDOUT:   %require_complete.loc18_34 => constants.%complete_type.091
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.a46
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0c3
// CHECK:STDOUT:   %.Self.as_type.loc18_34.2 => constants.%.Self.as_type.920
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc18_34.2 => constants.%.Self.as_wit.iface0.67d
// CHECK:STDOUT:   %impl.elem0.loc18_34.2 => constants.%impl.elem0.1bd
// CHECK:STDOUT:   %ptr.loc18_40.2 => constants.%ptr.235
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.52c
// CHECK:STDOUT:   %V.loc18_43.2 => bool
// CHECK:STDOUT:   %V.patt.loc18_43.2 => constants.%V.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_17 => constants.%complete_type.6bd
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Empty.type.d5a: type = generic_interface_type @Empty [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Empty.generic: %Empty.type.d5a = struct_value () [template]
// CHECK:STDOUT:   %Empty.type.a54: type = facet_type <@Empty, @Empty(%W)> [symbolic]
// CHECK:STDOUT:   %Self: %Empty.type.a54 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %assoc_type.263: type = assoc_entity_type %Empty.type.a54, type [symbolic]
// CHECK:STDOUT:   %assoc0.78a: %assoc_type.263 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.f74: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.ec8: %Empty.type.f74 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %require_complete.9fb: <witness> = require_complete_type %Empty.type.f74 [symbolic]
// CHECK:STDOUT:   %assoc_type.c52: type = assoc_entity_type %Empty.type.f74, type [symbolic]
// CHECK:STDOUT:   %assoc0.292: %assoc_type.c52 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.b86: <witness> = facet_access_witness %.Self.ec8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.76f: type = impl_witness_access %.Self.as_wit.b86, element0 [symbolic]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Empty_where.type.963: type = facet_type <@Empty, @Empty(%T) where %impl.elem0.76f = %ptr.251> [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.067: <witness> = require_complete_type %Empty_where.type.963 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Empty.type.4af: type = facet_type <@Empty, @Empty(%i32)> [template]
// CHECK:STDOUT:   %.Self.a46: %Empty.type.4af = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type.488: type = assoc_entity_type %Empty.type.4af, type [template]
// CHECK:STDOUT:   %assoc0.7bc: %assoc_type.488 = assoc_entity element0, @Empty.%A [template]
// CHECK:STDOUT:   %.Self.as_wit.c5e: <witness> = facet_access_witness %.Self.a46 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b4f: type = impl_witness_access %.Self.as_wit.c5e, element0 [symbolic]
// CHECK:STDOUT:   %ptr.dc3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %Empty_where.type.5d3: type = facet_type <@Empty, @Empty(%i32) where %impl.elem0.b4f = %ptr.dc3> [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %complete_type.6c0: <witness> = complete_type_witness %Empty.type.4af [template]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H, @H(%i32, bool) [template]
// CHECK:STDOUT:   %complete_type.a60: <witness> = complete_type_witness %Empty_where.type.5d3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: %Empty.type.d5a = interface_decl @Empty [template = constants.%Empty.generic] {
// CHECK:STDOUT:     %W.patt.loc11_17.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc11_17.1, runtime_param<invalid> [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc11_17.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %T.patt.loc18_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc18_6.1, runtime_param<invalid> [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt: @H.%Empty_where.type (%Empty_where.type.963) = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: @H.%Empty_where.type (%Empty_where.type.963) = value_param_pattern %U.patt, runtime_param0
// CHECK:STDOUT:     %V.patt.loc18_43.1: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc18_43.1, runtime_param<invalid> [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc18_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.963) = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Empty_where.type (constants.%Empty_where.type.963)] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [template = constants.%Empty.generic]
// CHECK:STDOUT:       %T.ref.loc18_25: type = name_ref T, %T.loc18_6.1 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:       %Empty.type.loc18_26.1: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc18_26.2 (constants.%Empty.type.f74)]
// CHECK:STDOUT:       %.Self.1: @H.%Empty.type.loc18_26.2 (%Empty.type.f74) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.ec8)]
// CHECK:STDOUT:       %.Self.ref: @H.%Empty.type.loc18_26.2 (%Empty.type.f74) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.ec8)]
// CHECK:STDOUT:       %.loc18_34: @H.%assoc_type (%assoc_type.c52) = specific_constant @Empty.%assoc0.loc12_15.1, @Empty(constants.%T) [symbolic = %assoc0 (constants.%assoc0.292)]
// CHECK:STDOUT:       %A.ref: @H.%assoc_type (%assoc_type.c52) = name_ref A, %.loc18_34 [symbolic = %assoc0 (constants.%assoc0.292)]
// CHECK:STDOUT:       %.Self.as_wit.loc18_34.1: <witness> = facet_access_witness %.Self.ref [symbolic = %.Self.as_wit.loc18_34.2 (constants.%.Self.as_wit.b86)]
// CHECK:STDOUT:       %impl.elem0.loc18_34.1: type = impl_witness_access %.Self.as_wit.loc18_34.1, element0 [symbolic = %impl.elem0.loc18_34.2 (constants.%impl.elem0.76f)]
// CHECK:STDOUT:       %T.ref.loc18_39: type = name_ref T, %T.loc18_6.1 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc18_40.1: type = ptr_type %T [symbolic = %ptr.loc18_40.2 (constants.%ptr.251)]
// CHECK:STDOUT:       %.loc18_28.2: type = where_expr %.Self.1 [symbolic = %Empty_where.type (constants.%Empty_where.type.963)] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_34.1, %ptr.loc18_40.1
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.963) = bind_name U, %U.param
// CHECK:STDOUT:     %V.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc18_43.1: type = bind_symbolic_name V, 1, %V.param [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %U.patt: %Empty_where.type.5d3 = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: %Empty_where.type.5d3 = value_param_pattern %U.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: %Empty_where.type.5d3 = value_param runtime_param0
// CHECK:STDOUT:     %.loc20_20.1: type = splice_block %.loc20_20.2 [template = constants.%Empty_where.type.5d3] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d5a = name_ref Empty, file.%Empty.decl [template = constants.%Empty.generic]
// CHECK:STDOUT:       %int_32.loc20_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %Empty.type: type = facet_type <@Empty, @Empty(constants.%i32)> [template = constants.%Empty.type.4af]
// CHECK:STDOUT:       %.Self: %Empty.type.4af = bind_symbolic_name .Self [symbolic = constants.%.Self.a46]
// CHECK:STDOUT:       %.Self.ref: %Empty.type.4af = name_ref .Self, %.Self [symbolic = constants.%.Self.a46]
// CHECK:STDOUT:       %.loc20_26: %assoc_type.488 = specific_constant @Empty.%assoc0.loc12_15.1, @Empty(constants.%i32) [template = constants.%assoc0.7bc]
// CHECK:STDOUT:       %A.ref: %assoc_type.488 = name_ref A, %.loc20_26 [template = constants.%assoc0.7bc]
// CHECK:STDOUT:       %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit.c5e]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0.b4f]
// CHECK:STDOUT:       %int_32.loc20_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_31: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %ptr: type = ptr_type %i32 [template = constants.%ptr.dc3]
// CHECK:STDOUT:       %.loc20_20.2: type = where_expr %.Self [template = constants.%Empty_where.type.5d3] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0, %ptr
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: %Empty_where.type.5d3 = bind_name U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Empty(%W.loc11_17.1: type) {
// CHECK:STDOUT:   %W.loc11_17.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_17.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty, @Empty(%W.loc11_17.2)> [symbolic = %Empty.type (constants.%Empty.type.a54)]
// CHECK:STDOUT:   %Self.2: %Empty.type.a54 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type @Empty.%Empty.type (%Empty.type.a54), type [symbolic = %assoc_type (constants.%assoc_type.263)]
// CHECK:STDOUT:   %assoc0.loc12_15.2: @Empty.%assoc_type (%assoc_type.263) = assoc_entity element0, %A [symbolic = %assoc0.loc12_15.2 (constants.%assoc0.78a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Empty.%Empty.type (%Empty.type.a54) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %A: type = assoc_const_decl A [template]
// CHECK:STDOUT:     %assoc0.loc12_15.1: @Empty.%assoc_type (%assoc_type.263) = assoc_entity element0, %A [symbolic = %assoc0.loc12_15.2 (constants.%assoc0.78a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .A = %assoc0.loc12_15.1
// CHECK:STDOUT:     witness = (%A)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc18_6.1: type, %V.loc18_43.1: type) {
// CHECK:STDOUT:   %T.loc18_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc18_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Empty.type.loc18_26.2: type = facet_type <@Empty, @Empty(%T.loc18_6.2)> [symbolic = %Empty.type.loc18_26.2 (constants.%Empty.type.f74)]
// CHECK:STDOUT:   %.Self.2: @H.%Empty.type.loc18_26.2 (%Empty.type.f74) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.ec8)]
// CHECK:STDOUT:   %require_complete.loc18_34: <witness> = require_complete_type @H.%Empty.type.loc18_26.2 (%Empty.type.f74) [symbolic = %require_complete.loc18_34 (constants.%require_complete.9fb)]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type @H.%Empty.type.loc18_26.2 (%Empty.type.f74), type [symbolic = %assoc_type (constants.%assoc_type.c52)]
// CHECK:STDOUT:   %assoc0: @H.%assoc_type (%assoc_type.c52) = assoc_entity element0, @Empty.%A [symbolic = %assoc0 (constants.%assoc0.292)]
// CHECK:STDOUT:   %.Self.as_wit.loc18_34.2: <witness> = facet_access_witness %.Self.2 [symbolic = %.Self.as_wit.loc18_34.2 (constants.%.Self.as_wit.b86)]
// CHECK:STDOUT:   %impl.elem0.loc18_34.2: type = impl_witness_access %.Self.as_wit.loc18_34.2, element0 [symbolic = %impl.elem0.loc18_34.2 (constants.%impl.elem0.76f)]
// CHECK:STDOUT:   %ptr.loc18_40.2: type = ptr_type @H.%T.loc18_6.2 (%T) [symbolic = %ptr.loc18_40.2 (constants.%ptr.251)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc18_6.2) where %impl.elem0.loc18_34.2 (constants.%impl.elem0.76f) = %ptr.loc18_40.2 (constants.%ptr.251)> [symbolic = %Empty_where.type (constants.%Empty_where.type.963)]
// CHECK:STDOUT:   %V.loc18_43.2: type = bind_symbolic_name V, 1 [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc18_43.2: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_17: <witness> = require_complete_type @H.%Empty_where.type (%Empty_where.type.963) [symbolic = %require_complete.loc18_17 (constants.%require_complete.067)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %U.param_patt: @H.%Empty_where.type (%Empty_where.type.963), %V.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%U.param_patt: %Empty_where.type.5d3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [template = constants.%H]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %U.ref: %Empty_where.type.5d3 = name_ref U, %U
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc21_17.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc21_17.2: type = converted %bool.make_type, %.loc21_17.1 [template = bool]
// CHECK:STDOUT:   %H.specific_fn: <specific function> = specific_function %H.ref, @H(constants.%i32, bool) [template = constants.%H.specific_fn]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %H.specific_fn(%U.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%W) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(%W.loc11_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.f74
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %assoc_type => constants.%assoc_type.c52
// CHECK:STDOUT:   %assoc0.loc12_15.2 => constants.%assoc0.292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%T
// CHECK:STDOUT:   %Empty.type.loc18_26.2 => constants.%Empty.type.f74
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.ec8
// CHECK:STDOUT:   %require_complete.loc18_34 => constants.%require_complete.9fb
// CHECK:STDOUT:   %assoc_type => constants.%assoc_type.c52
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.292
// CHECK:STDOUT:   %.Self.as_wit.loc18_34.2 => constants.%.Self.as_wit.b86
// CHECK:STDOUT:   %impl.elem0.loc18_34.2 => constants.%impl.elem0.76f
// CHECK:STDOUT:   %ptr.loc18_40.2 => constants.%ptr.251
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.963
// CHECK:STDOUT:   %V.loc18_43.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc18_43.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(@H.%T.loc18_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%i32) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%i32
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.4af
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %assoc_type => constants.%assoc_type.488
// CHECK:STDOUT:   %assoc0.loc12_15.2 => constants.%assoc0.7bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32, bool) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%i32
// CHECK:STDOUT:   %Empty.type.loc18_26.2 => constants.%Empty.type.4af
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.a46
// CHECK:STDOUT:   %require_complete.loc18_34 => constants.%complete_type.6c0
// CHECK:STDOUT:   %assoc_type => constants.%assoc_type.488
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7bc
// CHECK:STDOUT:   %.Self.as_wit.loc18_34.2 => constants.%.Self.as_wit.c5e
// CHECK:STDOUT:   %impl.elem0.loc18_34.2 => constants.%impl.elem0.b4f
// CHECK:STDOUT:   %ptr.loc18_40.2 => constants.%ptr.dc3
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.5d3
// CHECK:STDOUT:   %V.loc18_43.2 => bool
// CHECK:STDOUT:   %V.patt.loc18_43.2 => bool
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_17 => constants.%complete_type.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:

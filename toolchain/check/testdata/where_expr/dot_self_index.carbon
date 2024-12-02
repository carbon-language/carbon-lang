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
// CHECK:STDOUT:   %Empty.type.1: type = generic_interface_type @Empty [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Empty: %Empty.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Empty.type.2: type = facet_type <@Empty, @Empty(%W)> [symbolic]
// CHECK:STDOUT:   %Self: %Empty.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Empty.type.2, type [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.3: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.1: %Empty.type.3 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Empty.type.3, type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %.5: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %.6: type = interface_witness_access %.5, element0 [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Empty_where.type.1: type = facet_type <@Empty, @Empty(%T) where %.6 = %.7> [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 1 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.8 [template]
// CHECK:STDOUT:   %Empty.type.4: type = facet_type <@Empty, @Empty(%i32)> [template]
// CHECK:STDOUT:   %.Self.2: %Empty.type.4 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %Empty.type.4, type [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, @Empty.%A [template]
// CHECK:STDOUT:   %.11: <witness> = facet_access_witness %.Self.2 [symbolic]
// CHECK:STDOUT:   %.12: type = interface_witness_access %.11, element0 [symbolic]
// CHECK:STDOUT:   %.13: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %Empty_where.type.2: type = facet_type <@Empty, @Empty(%i32) where %.12 = %.13> [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %.14: <specific function> = specific_function %H, @H(%i32, bool) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
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
// CHECK:STDOUT:   %Empty.decl: %Empty.type.1 = interface_decl @Empty [template = constants.%Empty] {
// CHECK:STDOUT:     %W.patt.loc11_17.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc11_17.1, runtime_param<invalid> [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc11_17.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %T.patt.loc18_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc18_6.1, runtime_param<invalid> [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt: @H.%Empty_where.type (%Empty_where.type.1) = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: @H.%Empty_where.type (%Empty_where.type.1) = value_param_pattern %U.patt, runtime_param0
// CHECK:STDOUT:     %V.patt.loc18_43.1: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc18_43.1, runtime_param<invalid> [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Empty.ref: %Empty.type.1 = name_ref Empty, file.%Empty.decl [template = constants.%Empty]
// CHECK:STDOUT:     %T.ref.loc18_25: type = name_ref T, %T.loc18_6.1 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:     %Empty.type.loc18_24.1: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc18_24.2 (constants.%Empty.type.3)]
// CHECK:STDOUT:     %.Self.1: %Empty.type.3 = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %Empty.type.3 = name_ref .Self, %.Self.1 [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.loc18_34.1: %.3 = specific_constant @Empty.%.loc12_15.1, @Empty(constants.%T) [symbolic = constants.%.4]
// CHECK:STDOUT:     %A.ref: %.3 = name_ref A, %.loc18_34.1 [symbolic = constants.%.4]
// CHECK:STDOUT:     %.loc18_34.2: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.5]
// CHECK:STDOUT:     %.loc18_34.3: type = interface_witness_access %.loc18_34.2, element0 [symbolic = constants.%.6]
// CHECK:STDOUT:     %T.ref.loc18_39: type = name_ref T, %T.loc18_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc18_40: type = ptr_type %T [symbolic = constants.%.7]
// CHECK:STDOUT:     %.loc18_28.1: type = where_expr %.Self.1 [symbolic = %Empty_where.type (constants.%Empty_where.type.1)] {
// CHECK:STDOUT:       requirement_rewrite %.loc18_34.3, %.loc18_40
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc18_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.1) = value_param runtime_param0
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.1) = bind_name U, %U.param
// CHECK:STDOUT:     %V.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc18_43.1: type = bind_symbolic_name V, 1, %V.param [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %U.patt: %Empty_where.type.2 = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: %Empty_where.type.2 = value_param_pattern %U.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Empty.ref: %Empty.type.1 = name_ref Empty, file.%Empty.decl [template = constants.%Empty]
// CHECK:STDOUT:     %.loc20_15: Core.IntLiteral = int_value 32 [template = constants.%.8]
// CHECK:STDOUT:     %int.make_type_signed.loc20_15: init type = call constants.%Int(%.loc20_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_14.1: type = value_of_initializer %int.make_type_signed.loc20_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_14.2: type = converted %int.make_type_signed.loc20_15, %.loc20_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %Empty.type: type = facet_type <@Empty, @Empty(constants.%i32)> [template = constants.%Empty.type.4]
// CHECK:STDOUT:     %.Self: %Empty.type.4 = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: %Empty.type.4 = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.loc20_26.1: %.9 = specific_constant @Empty.%.loc12_15.1, @Empty(constants.%i32) [template = constants.%.10]
// CHECK:STDOUT:     %A.ref: %.9 = name_ref A, %.loc20_26.1 [template = constants.%.10]
// CHECK:STDOUT:     %.loc20_26.2: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.11]
// CHECK:STDOUT:     %.loc20_26.3: type = interface_witness_access %.loc20_26.2, element0 [symbolic = constants.%.12]
// CHECK:STDOUT:     %.loc20_31: Core.IntLiteral = int_value 32 [template = constants.%.8]
// CHECK:STDOUT:     %int.make_type_signed.loc20_31: init type = call constants.%Int(%.loc20_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_34.1: type = value_of_initializer %int.make_type_signed.loc20_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_34.2: type = converted %int.make_type_signed.loc20_31, %.loc20_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc20_34.3: type = ptr_type %i32 [template = constants.%.13]
// CHECK:STDOUT:     %.loc20_20: type = where_expr %.Self [template = constants.%Empty_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %.loc20_26.3, %.loc20_34.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %Empty_where.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %U: %Empty_where.type.2 = bind_name U, %U.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Empty(%W.loc11_17.1: type) {
// CHECK:STDOUT:   %W.loc11_17.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc11_17.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc11_17.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc11_17.2 (constants.%W.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty, @Empty(%W.loc11_17.2)> [symbolic = %Empty.type (constants.%Empty.type.2)]
// CHECK:STDOUT:   %Self.2: %Empty.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %.loc12_15.2: type = assoc_entity_type @Empty.%Empty.type (%Empty.type.2), type [symbolic = %.loc12_15.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc12_15.3: @Empty.%.loc12_15.2 (%.1) = assoc_entity element0, %A [symbolic = %.loc12_15.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Empty.%Empty.type (%Empty.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %A: type = assoc_const_decl A [template]
// CHECK:STDOUT:     %.loc12_15.1: @Empty.%.loc12_15.2 (%.1) = assoc_entity element0, %A [symbolic = %.loc12_15.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .A = %.loc12_15.1
// CHECK:STDOUT:     witness = (%A)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc18_6.1: type, %V.loc18_43.1: type) {
// CHECK:STDOUT:   %T.loc18_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc18_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Empty.type.loc18_24.2: type = facet_type <@Empty, @Empty(%T.loc18_6.2)> [symbolic = %Empty.type.loc18_24.2 (constants.%Empty.type.3)]
// CHECK:STDOUT:   %.Self.loc18: @H.%Empty.type.loc18_24.2 (%Empty.type.3) = bind_symbolic_name .Self [symbolic = %.Self.loc18 (constants.%.Self.1)]
// CHECK:STDOUT:   %.loc18_28.2: <witness> = facet_access_witness %.Self.loc18 [symbolic = %.loc18_28.2 (constants.%.5)]
// CHECK:STDOUT:   %.loc18_28.3: type = interface_witness_access %.loc18_28.2, element0 [symbolic = %.loc18_28.3 (constants.%.6)]
// CHECK:STDOUT:   %.loc18_28.4: type = ptr_type @H.%T.loc18_6.2 (%T) [symbolic = %.loc18_28.4 (constants.%.7)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc18_6.2) where %.loc18_28.3 (constants.%.6) = %.loc18_28.4 (constants.%.7)> [symbolic = %Empty_where.type (constants.%Empty_where.type.1)]
// CHECK:STDOUT:   %V.loc18_43.2: type = bind_symbolic_name V, 1 [symbolic = %V.loc18_43.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc18_43.2: type = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc18_43.2 (constants.%V.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %U.param_patt: @H.%Empty_where.type (%Empty_where.type.1), %V.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%U.param_patt: %Empty_where.type.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [template = constants.%H]
// CHECK:STDOUT:   %.loc21_5: Core.IntLiteral = int_value 32 [template = constants.%.8]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_5) [template = constants.%i32]
// CHECK:STDOUT:   %U.ref: %Empty_where.type.2 = name_ref U, %U
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc21_4.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_4.2: type = converted %int.make_type_signed.loc21, %.loc21_4.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_4.3: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc21_4.4: type = converted %bool.make_type, %.loc21_4.3 [template = bool]
// CHECK:STDOUT:   %.loc21_3: <specific function> = specific_function %H.ref, @H(constants.%i32, bool) [template = constants.%.14]
// CHECK:STDOUT:   %H.call: init %empty_tuple.type = call %.loc21_3(%U.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%W) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(%W.loc11_17.2) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %.loc12_15.2 => constants.%.3
// CHECK:STDOUT:   %.loc12_15.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(@H.%T.loc18_6.2) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%T
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%T
// CHECK:STDOUT:   %Empty.type.loc18_24.2 => constants.%Empty.type.3
// CHECK:STDOUT:   %.Self.loc18 => constants.%.Self.1
// CHECK:STDOUT:   %.loc18_28.2 => constants.%.5
// CHECK:STDOUT:   %.loc18_28.3 => constants.%.6
// CHECK:STDOUT:   %.loc18_28.4 => constants.%.7
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.1
// CHECK:STDOUT:   %V.loc18_43.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc18_43.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%i32) {
// CHECK:STDOUT:   %W.loc11_17.2 => constants.%i32
// CHECK:STDOUT:   %W.patt.loc11_17.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %.loc12_15.2 => constants.%.9
// CHECK:STDOUT:   %.loc12_15.3 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32, bool) {
// CHECK:STDOUT:   %T.loc18_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc18_6.2 => constants.%i32
// CHECK:STDOUT:   %Empty.type.loc18_24.2 => constants.%Empty.type.4
// CHECK:STDOUT:   %.Self.loc18 => constants.%.Self.2
// CHECK:STDOUT:   %.loc18_28.2 => constants.%.11
// CHECK:STDOUT:   %.loc18_28.3 => constants.%.12
// CHECK:STDOUT:   %.loc18_28.4 => constants.%.13
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.2
// CHECK:STDOUT:   %V.loc18_43.2 => bool
// CHECK:STDOUT:   %V.patt.loc18_43.2 => bool
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/no_prelude/import.carbon

// --- core.carbon

// A minimized prelude intended to contain as few functions as possible.
package Core;

fn IntLiteral() -> type = "int_literal.make_type";

private fn MakeInt(size: IntLiteral()) -> type = "int.make_type_signed";

class Int(N:! IntLiteral()) {
  adapt MakeInt(N);
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl forall [N:! IntLiteral()] IntLiteral() as ImplicitAs(Int(N)) {
  fn Convert[self: Self]() -> Int(N) = "int.convert_checked";
}

// --- implicit.carbon

package Implicit;

import Core;

class C(S:! {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});
// var d: C({.a = 1, .b = 2}) = F();

// --- implicit.impl.carbon

impl package Implicit;

var c: C({.a = 1, .b = 2}) = F();

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %MakeInt.type: type = fn_type @MakeInt [template]
// CHECK:STDOUT:   %MakeInt: %MakeInt.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [template]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %ImplicitAs.type.3 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%Int) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2 = facet_value Core.IntLiteral, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .MakeInt [private] = %MakeInt.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeInt.decl: %MakeInt.type = fn_decl @MakeInt [template = constants.%MakeInt] {
// CHECK:STDOUT:     %size.patt: Core.IntLiteral = binding_pattern size
// CHECK:STDOUT:     %size.param_patt: Core.IntLiteral = value_param_pattern %size.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %size.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_37.1: type = splice_block %.loc7_37.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_37.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_37.3: type = converted %int_literal.make_type, %.loc7_37.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %size: Core.IntLiteral = bind_name size, %size.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = class_decl @Int [template = constants.%Int.generic] {
// CHECK:STDOUT:     %N.patt.loc9_11.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc9_11.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc9_11.1, runtime_param<invalid> [symbolic = %N.patt.loc9_11.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc9_26.1: type = splice_block %.loc9_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_26.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc9_26.3: type = converted %int_literal.make_type, %.loc9_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc9_11.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc9_11.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc13_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_22.1, runtime_param<invalid> [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %N.patt.loc17_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc17_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc17_14.1, runtime_param<invalid> [symbolic = %N.patt.loc17_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc17_32: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc17_43: init type = call %IntLiteral.ref.loc17_32() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc17_43.1: type = value_of_initializer %int_literal.make_type.loc17_43 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc17_43.2: type = converted %int_literal.make_type.loc17_43, %.loc17_43.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %Int.ref: %Int.type = name_ref Int, file.%Int.decl [template = constants.%Int.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc17_14.1 [symbolic = %N.loc17_14.2 (constants.%N)]
// CHECK:STDOUT:     %Int.loc17_64.1: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc17_64.2 (constants.%Int)]
// CHECK:STDOUT:     %ImplicitAs.type.loc17_65.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%Int)> [symbolic = %ImplicitAs.type.loc17_65.2 (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc17_29.1: type = splice_block %.loc17_29.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref.loc17_18: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc17_29: init type = call %IntLiteral.ref.loc17_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc17_29.2: type = value_of_initializer %int_literal.make_type.loc17_29 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc17_29.3: type = converted %int_literal.make_type.loc17_29, %.loc17_29.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc17_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc17_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl), @impl(constants.%N) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc13_22.1: type) {
// CHECK:STDOUT:   %T.loc13_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc14_32.2: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_32.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc14_20.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc14_20.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc13_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc14_20.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc14_20.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_32.1: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_32.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc14_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%N.loc17_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc17_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc17_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc17_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc17_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   %Int.loc17_64.2: type = class_type @Int, @Int(%N.loc17_14.2) [symbolic = %Int.loc17_64.2 (constants.%Int)]
// CHECK:STDOUT:   %ImplicitAs.type.loc17_65.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.loc17_64.2)> [symbolic = %ImplicitAs.type.loc17_65.2 (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type @impl.%ImplicitAs.type.loc17_65.2 (%ImplicitAs.type.3) [symbolic = %require_complete.loc17 (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.decl), @impl(%N.loc17_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl(%N.loc17_14.2) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type @impl.%Int.loc17_64.2 (%Int) [symbolic = %require_complete.loc18 (constants.%require_complete.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc17_43.2 as %ImplicitAs.type.loc17_65.1 {
// CHECK:STDOUT:     %Convert.decl: @impl.%Convert.type (%Convert.type.3) = fn_decl @Convert.2 [symbolic = @impl.%Convert (constants.%Convert.3)] {
// CHECK:STDOUT:       %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Int.loc18_36.1 (%Int) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Int.loc18_36.1 (%Int) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Int.ref: %Int.type = name_ref Int, file.%Int.decl [template = constants.%Int.generic]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, @impl.%N.loc17_14.1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:       %Int.loc18_36.2: type = class_type @Int, @Int(constants.%N) [symbolic = %Int.loc18_36.1 (constants.%Int)]
// CHECK:STDOUT:       %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%.loc17_43.2 [template = Core.IntLiteral]
// CHECK:STDOUT:       %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Int.loc18_36.1 (%Int) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Int.loc18_36.1 (%Int) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Convert = %Convert.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Int(%N.loc9_11.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc9_11.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc9_11.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc9_11.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc9_11.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N.loc9_11.2 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Int.%iN.builtin (%iN.builtin) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @Int.%iN.builtin (%iN.builtin) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %MakeInt.ref: %MakeInt.type = name_ref MakeInt, file.%MakeInt.decl [template = constants.%MakeInt]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc9_11.1 [symbolic = %N.loc9_11.2 (constants.%N)]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %MakeInt.ref(%N.ref) [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc10_19.1: type = value_of_initializer %int.make_type_signed [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %int.make_type_signed, %.loc10_19.1 [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:     adapt_decl %.loc10_19.2 [template]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %iN.builtin [symbolic = %complete_type.loc11_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Int
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInt(%size.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc14_20.1 (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@impl.%N.loc17_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int.loc18_36.1: type = class_type @Int, @Int(%N) [symbolic = %Int.loc18_36.1 (constants.%Int)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%Int.loc18_36.1 (%Int) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%N) {
// CHECK:STDOUT:   %N.loc9_11.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc9_11.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc13_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Int) {
// CHECK:STDOUT:   %T.loc13_22.2 => constants.%Int
// CHECK:STDOUT:   %T.patt.loc13_22.2 => constants.%Int
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0.loc14_32.2 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%N) {
// CHECK:STDOUT:   %N.loc17_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc17_14.2 => constants.%N
// CHECK:STDOUT:   %Int.loc17_64.2 => constants.%Int
// CHECK:STDOUT:   %ImplicitAs.type.loc17_65.2 => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %require_complete.loc17 => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@impl.%N.loc17_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.%Int.loc17_64.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%N.loc17_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int.loc18_36.1 => constants.%Int
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@Convert.2.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Int, constants.%ImplicitAs.facet) {
// CHECK:STDOUT:   %T => constants.%Int
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b.1 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.1 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.1: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.9 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, imports.%import_ref.9 [template]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.10 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %ImplicitAs.type.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (imports.%import_ref.14), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %ImplicitAs.type.3 [template]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (imports.%import_ref.14), @impl(%int_32) [template]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (imports.%import_ref.15), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @impl(%int_32) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Core//default, loc11_1, loaded [symbolic = @Int.%complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//default, inst53 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst80 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.7: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = import_ref Core//default, loc14_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Core//default, loc14_32, unloaded
// CHECK:STDOUT:   %import_ref.11: <witness> = import_ref Core//default, loc17_67, loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//default, loc17_43, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//default, loc17_65, loaded [symbolic = @impl.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.14: @impl.%Convert.type (%Convert.type.3) = import_ref Core//default, loc18_38, loaded [symbolic = @impl.%Convert (constants.%Convert.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %S.patt.loc6_9.1: %struct_type.a.b.1 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc6_9.2 (constants.%S.patt)]
// CHECK:STDOUT:     %S.param_patt: %struct_type.a.b.1 = value_param_pattern %S.patt.loc6_9.1, runtime_param<invalid> [symbolic = %S.patt.loc6_9.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %S.param: %struct_type.a.b.1 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6: type = splice_block %struct_type.a.b [template = constants.%struct_type.a.b.1] {
// CHECK:STDOUT:       %int_32.loc6_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %int_32.loc6_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc6_9.1: %struct_type.a.b.1 = bind_symbolic_name S, 0, %S.param [symbolic = %S.loc6_9.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:     %.loc7_28.1: %struct_type.a.b.2 = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc7_28.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:     %Convert.bound.loc7_28.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_28.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:     %Convert.specific_fn.loc7_28.1: <specific function> = specific_function %Convert.bound.loc7_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:     %int.convert_checked.loc7_28.1: init %i32 = call %Convert.specific_fn.loc7_28.1(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:     %.loc7_28.2: %i32 = value_of_initializer %int.convert_checked.loc7_28.1 [template = constants.%int_1.2]
// CHECK:STDOUT:     %.loc7_28.3: %i32 = converted %int_1, %.loc7_28.2 [template = constants.%int_1.2]
// CHECK:STDOUT:     %impl.elem0.loc7_28.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.2, element0 [template = constants.%Convert.4]
// CHECK:STDOUT:     %Convert.bound.loc7_28.2: <bound method> = bound_method %int_2, %impl.elem0.loc7_28.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:     %Convert.specific_fn.loc7_28.2: <specific function> = specific_function %Convert.bound.loc7_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:     %int.convert_checked.loc7_28.2: init %i32 = call %Convert.specific_fn.loc7_28.2(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc7_28.4: %i32 = value_of_initializer %int.convert_checked.loc7_28.2 [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc7_28.5: %i32 = converted %int_2, %.loc7_28.4 [template = constants.%int_2.2]
// CHECK:STDOUT:     %struct: %struct_type.a.b.1 = struct_value (%.loc7_28.3, %.loc7_28.5) [template = constants.%struct]
// CHECK:STDOUT:     %.loc7_29: %struct_type.a.b.1 = converted %.loc7_28.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct) [template = constants.%C.2]
// CHECK:STDOUT:     %return.param: ref %C.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, imports.%import_ref.9 [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:     .Convert = imports.%import_ref.7
// CHECK:STDOUT:     witness = (imports.%import_ref.8)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(constants.%N: Core.IntLiteral) [from "core.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type @impl.%ImplicitAs.type (%ImplicitAs.type.4) [symbolic = %require_complete.1 (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.14), @impl(%N) [symbolic = %impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type @impl.%Int (%Int) [symbolic = %require_complete.2 (constants.%require_complete.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Int(constants.%N: Core.IntLiteral) [from "core.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Int.%iN.builtin (%iN.builtin) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Int.%iN.builtin (%iN.builtin) [symbolic = %complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc6_9.1: %struct_type.a.b.1) {
// CHECK:STDOUT:   %S.loc6_9.2: %struct_type.a.b.1 = bind_symbolic_name S, 0 [symbolic = %S.loc6_9.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc6_9.2: %struct_type.a.b.1 = symbolic_binding_pattern S, 0 [symbolic = %S.patt.loc6_9.2 (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.2) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%Int (%Int) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %complete_type => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %N.patt => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%i32.builtin
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %complete_type => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.loc6_9.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc6_9.2 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Int) {
// CHECK:STDOUT:   %T => constants.%Int
// CHECK:STDOUT:   %T.patt => constants.%Int
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %require_complete.1 => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@impl.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.%Int) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@Convert.2.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %N.patt => constants.%int_32
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S.loc6_9.2 => constants.%struct
// CHECK:STDOUT:   %S.patt.loc6_9.2 => constants.%struct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %iN.builtin [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %iN.builtin [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.1: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.12 [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %ImplicitAs.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (imports.%import_ref.13), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %Int [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %S: %struct_type.a.b.1 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.1 = symbolic_binding_pattern S, 0 [symbolic]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.2: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.3 [template]
// CHECK:STDOUT:   %assoc0.2: %Convert.assoc_type.2 = assoc_entity element0, imports.%import_ref.12 [template]
// CHECK:STDOUT:   %assoc0.3: %Convert.assoc_type.1 = assoc_entity element0, imports.%import_ref.17 [symbolic]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %ImplicitAs.type.4 [template]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (imports.%import_ref.13), @impl(%int_32) [template]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (imports.%import_ref.18), @impl(%N) [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @impl(%int_32) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%struct) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref Implicit//default, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.2: %F.type = import_ref Implicit//default, F, loaded [template = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.16
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Implicit//default, inst114 [indirect], loaded [symbolic = @impl.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:   %import_ref.5: <witness> = import_ref Implicit//default, inst31 [indirect], loaded [symbolic = @Int.%complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:   %import_ref.6 = import_ref Implicit//default, inst32 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Implicit//default, inst81 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.8: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = import_ref Implicit//default, inst82 [indirect], loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Implicit//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.10: type = import_ref Implicit//default, inst119 [indirect], loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Implicit//default, inst120 [indirect], loaded [symbolic = @impl.%ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %import_ref.12 = import_ref Implicit//default, inst95 [indirect], unloaded
// CHECK:STDOUT:   %import_ref.13: @impl.%Convert.type (%Convert.type.2) = import_ref Implicit//default, inst122 [indirect], loaded [symbolic = @impl.%Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %import_ref.14: <witness> = import_ref Implicit//default, loc6_34, loaded [template = constants.%complete_type.2]
// CHECK:STDOUT:   %import_ref.15 = import_ref Implicit//default, inst57 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .F = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %c.var: ref %C.2 = var c
// CHECK:STDOUT:   %c: ref %C.2 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "implicit.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.1)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.1) = assoc_entity element0, imports.%import_ref.12 [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7
// CHECK:STDOUT:     .Convert = imports.%import_ref.8
// CHECK:STDOUT:     witness = (imports.%import_ref.9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(constants.%N: Core.IntLiteral) [from "implicit.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3)]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type @impl.%ImplicitAs.type (%ImplicitAs.type.3) [symbolic = %require_complete.1 (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.13), @impl(%N) [symbolic = %impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl(%N) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @impl.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type @impl.%Int (%Int) [symbolic = %require_complete.2 (constants.%require_complete.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.10 as imports.%import_ref.11 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Int(constants.%N: Core.IntLiteral) [from "implicit.carbon"] {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %N [symbolic = %iN.builtin (constants.%iN.builtin)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Int.%iN.builtin (%iN.builtin) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Int.%iN.builtin (%iN.builtin) [symbolic = %complete_type (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.6
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%S: %struct_type.a.b.1) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.1 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %S.patt: %struct_type.a.b.1 = symbolic_binding_pattern S, 0 [symbolic = %S.patt (constants.%S.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.15
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self: %ImplicitAs.type.2) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic = %Int (constants.%Int)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%Int (%Int) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref.2 [template = constants.%F]
// CHECK:STDOUT:   %.loc4: ref %C.2 = splice_block file.%c.var {}
// CHECK:STDOUT:   %F.call: init %C.2 = call %F.ref() to %.loc4
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%iN.builtin
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %complete_type => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Int) {
// CHECK:STDOUT:   %T => constants.%Int
// CHECK:STDOUT:   %T.patt => constants.%Int
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %require_complete.1 => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@impl.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.%Int) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(@Convert.2.%N) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %Int => constants.%Int
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Int(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %N.patt => constants.%int_32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %iN.builtin => constants.%i32.builtin
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %complete_type => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %S.patt => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %N.patt => constants.%int_32
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%int_32) {
// CHECK:STDOUT:   %N => constants.%int_32
// CHECK:STDOUT:   %Int => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:   %S.patt => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

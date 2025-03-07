// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/no_prelude/fail_todo_call_combined_impl_witness.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/no_prelude/fail_todo_call_combined_impl_witness.carbon

// --- core.carbon
package Core;

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

impl forall [T:! type] T as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}

// --- fail_todo_call_combined_impl_witness.carbon

library "[[@TEST_NAME]]";

import Core;

interface Empty {
}
interface A {
  fn AA();
}
interface B {
  fn BB();
}

class C {}
impl C as Empty {}
impl C as A {
  fn AA() {}
}
impl C as B {
  fn BB() {}
}

fn G[T:! A & Empty & B](t: T) {
  t.AA();
  t.BB();

  T.AA();
  T.BB();

  // TODO: Qualified lookup of `AA` and `BB` should also be possible here, using
  // the witnesses found in type deduction.

  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+7]]:3: error: cannot implicitly convert value of type `Empty & A & B` to `A` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   T.(A.AA)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+4]]:3: note: type `Empty & A & B` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   T.(A.AA)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  T.(A.AA)();
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+10]]:3: error: cannot implicitly convert value of type `Empty & A & B` to `B` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   T.(B.BB)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+7]]:3: note: type `Empty & A & B` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   T.(B.BB)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE-24]]:3: error: semantics TODO: `incorrect witness for multiple interfaces in a facet type` [SemanticsTodo]
  // CHECK:STDERR:   t.AA();
  // CHECK:STDERR:   ^~~~
  T.(B.BB)();
}

fn F() {
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+11]]:3: note: in `G(C as Empty & A & B)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE-34]]:3: error: semantics TODO: `incorrect witness for multiple interfaces in a facet type` [SemanticsTodo]
  // CHECK:STDERR:   t.BB();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_call_combined_impl_witness.carbon:[[@LINE+4]]:3: note: in `G(C as Empty & A & B)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  G({} as C);
}

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type %As.type.8ba [symbolic]
// CHECK:STDOUT:   %assoc0.ac5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type %ImplicitAs.type.07f [symbolic]
// CHECK:STDOUT:   %assoc0.a50: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.121: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc3_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc3_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc7_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc7_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc15_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc3_14.1: type) {
// CHECK:STDOUT:   %Dest.loc3_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc3_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc3_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc3_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc3_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc3_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8ba) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc3_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc4_20.1: type = splice_block %.loc4_20.3 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc4_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc4_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc4_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc4_20.3: type = converted %Self.ref, %Self.as_type.loc4_20.2 [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc4_35.2 (constants.%assoc0.ac5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc4_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc7_22.1: type) {
// CHECK:STDOUT:   %Dest.loc7_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc7_22.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc7_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc7_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc7_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc7_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc8_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc7_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc8_20.1: type = splice_block %.loc8_20.3 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc8_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc8_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc8_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc8_20.3: type = converted %Self.ref, %Self.as_type.loc8_20.2 [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc8_35.2 (constants.%assoc0.a50)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc8_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc12_37: type = facet_access_type %Self.ref.loc12_37 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc12_37: type = converted %Self.ref.loc12_37, %Self.as_type.loc12_37 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc12_15.1: type = splice_block %.loc12_15.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc12_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc12_15.2: type = facet_access_type %Self.ref.loc12_15 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc12_15.2: type = converted %Self.ref.loc12_15, %Self.as_type.loc12_15.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc12_28.1: type = splice_block %.loc12_28.2 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc12_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc12_28: type = facet_access_type %Self.ref.loc12_28 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc12_28.2: type = converted %Self.ref.loc12_28, %Self.as_type.loc12_28 [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.121]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T.loc15_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc15_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc15_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc15_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc16_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc16_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc16_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc3_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc4_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc4_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc7_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc8_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc8_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc12_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc12_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc3_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc3_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc4_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc3_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc7_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc7_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc8_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc7_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc12_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc12_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_combined_impl_witness.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.193: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.31d: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %AA.type.b97: type = fn_type @AA.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AA.c3a: %AA.type.b97 = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type %A.type [concrete]
// CHECK:STDOUT:   %assoc0.f33: %A.assoc_type = assoc_entity element0, @A.%AA.decl [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.783: %B.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BB.type.64d: type = fn_type @BB.1 [concrete]
// CHECK:STDOUT:   %BB.11b: %BB.type.64d = struct_value () [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type %B.type [concrete]
// CHECK:STDOUT:   %assoc0.019: %B.assoc_type = assoc_entity element0, @B.%BB.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %impl_witness.97a: <witness> = impl_witness (@impl.77d.%AA.decl) [concrete]
// CHECK:STDOUT:   %AA.type.c29: type = fn_type @AA.2 [concrete]
// CHECK:STDOUT:   %AA.95d: %AA.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet.afe: %A.type = facet_value %C, %impl_witness.97a [concrete]
// CHECK:STDOUT:   %impl_witness.fe4: <witness> = impl_witness (@impl.c94.%BB.decl) [concrete]
// CHECK:STDOUT:   %BB.type.24d: type = fn_type @BB.2 [concrete]
// CHECK:STDOUT:   %BB.fe8: %BB.type.24d = struct_value () [concrete]
// CHECK:STDOUT:   %B.facet.3a2: %B.type = facet_value %C, %impl_witness.fe4 [concrete]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.25f: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type %BitAnd.type [concrete]
// CHECK:STDOUT:   %assoc0.a63: %BitAnd.assoc_type = assoc_entity element0, imports.%Core.import_ref.a93 [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.ab9: %Op.type.27a = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_type.19f: type = facet_access_type %Self.25f [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.db8: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.f99: type = fn_type @Op.2, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.05a: %Op.type.f99 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %impl_witness.3ea: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(type) [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type.eb8: type = fn_type @Op.2, @impl.f92(type) [concrete]
// CHECK:STDOUT:   %Op.444: %Op.type.eb8 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, %impl_witness.3ea [concrete]
// CHECK:STDOUT:   %.d4d: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.bound.6b1: <bound method> = bound_method %A.type, %Op.444 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.444, @Op.2(type) [concrete]
// CHECK:STDOUT:   %bound_method.96c: <bound method> = bound_method %A.type, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.d5f: type = facet_type <@Empty & @A> [concrete]
// CHECK:STDOUT:   %Op.bound.b5b: <bound method> = bound_method %facet_type.d5f, %Op.444 [concrete]
// CHECK:STDOUT:   %bound_method.cb6: <bound method> = bound_method %facet_type.d5f, %Op.specific_fn [concrete]
// CHECK:STDOUT:   %facet_type.242: type = facet_type <@Empty & @A & @B> [concrete]
// CHECK:STDOUT:   %T.2df: %facet_type.242 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.fd9: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.2df [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.383: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T.2df [symbolic]
// CHECK:STDOUT:   %A.facet.d6d: %A.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.124: type = fn_type_with_self_type %AA.type.b97, %A.facet.d6d [symbolic]
// CHECK:STDOUT:   %impl.elem0.2b1: %.124 = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn.80f: <specific function> = specific_function %impl.elem0.2b1, @AA.1(%A.facet.d6d) [symbolic]
// CHECK:STDOUT:   %B.facet.1c2: %B.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.e3a: type = fn_type_with_self_type %BB.type.64d, %B.facet.1c2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f5b: %.e3a = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn.62f: <specific function> = specific_function %impl.elem0.f5b, @BB.1(%B.facet.1c2) [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.40a: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.837: type = assoc_entity_type %ImplicitAs.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc0.43db8b.1: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2d2: type = facet_type <@ImplicitAs, @ImplicitAs(%A.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.597: type = fn_type @Convert, @ImplicitAs(%A.type) [concrete]
// CHECK:STDOUT:   %Convert.d1e: %Convert.type.597 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.568: type = assoc_entity_type %ImplicitAs.type.2d2 [concrete]
// CHECK:STDOUT:   %assoc0.6fc: %ImplicitAs.assoc_type.568 = assoc_entity element0, imports.%Core.import_ref.207961.1 [concrete]
// CHECK:STDOUT:   %assoc0.43db8b.2: %ImplicitAs.assoc_type.837 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.8a4: type = facet_type <@ImplicitAs, @ImplicitAs(%B.type)> [concrete]
// CHECK:STDOUT:   %Convert.type.f1c: type = fn_type @Convert, @ImplicitAs(%B.type) [concrete]
// CHECK:STDOUT:   %Convert.e93: %Convert.type.f1c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.f97: type = assoc_entity_type %ImplicitAs.type.8a4 [concrete]
// CHECK:STDOUT:   %assoc0.19a: %ImplicitAs.assoc_type.f97 = assoc_entity element0, imports.%Core.import_ref.207961.1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.242 = facet_value %C, %impl_witness.1bc [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%facet_value) [concrete]
// CHECK:STDOUT:   %A.facet.beb: %A.type = facet_value %C, %impl_witness.1bc [concrete]
// CHECK:STDOUT:   %.778: type = fn_type_with_self_type %AA.type.b97, %A.facet.beb [concrete]
// CHECK:STDOUT:   %B.facet.5e0: %B.type = facet_value %C, %impl_witness.1bc [concrete]
// CHECK:STDOUT:   %.555: type = fn_type_with_self_type %BB.type.64d, %B.facet.5e0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAnd = %Core.BitAnd
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//default, inst100 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.08d: %BitAnd.assoc_type = import_ref Core//default, loc12_41, loaded [concrete = constants.%assoc0.a63]
// CHECK:STDOUT:   %Core.Op: %Op.type.27a = import_ref Core//default, Op, loaded [concrete = constants.%Op.ab9]
// CHECK:STDOUT:   %Core.import_ref.040: %BitAnd.type = import_ref Core//default, inst100 [no loc], loaded [symbolic = constants.%Self.25f]
// CHECK:STDOUT:   %Core.import_ref.51c: <witness> = import_ref Core//default, loc15_36, loaded [symbolic = @impl.f92.%impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//default, loc15_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//default, loc15_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.1e6: @impl.f92.%Op.type (%Op.type.f99) = import_ref Core//default, loc16_42, loaded [symbolic = @impl.f92.%Op (constants.%Op.05a)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc15_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = import_ref Core//default, loc8_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.43db8b.2)]
// CHECK:STDOUT:   %Core.Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//default, Convert, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//default, loc7_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst65 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc8_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.d78 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc16: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.77d [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (@impl.77d.%AA.decl) [concrete = constants.%impl_witness.97a]
// CHECK:STDOUT:   impl_decl @impl.c94 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (@impl.c94.%BB.decl) [concrete = constants.%impl_witness.fe4]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc24_6.1: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_6.2 (constants.%T.patt.fd9)]
// CHECK:STDOUT:     %t.patt: @G.%T.as_type.loc24_28.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @G.%T.as_type.loc24_28.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24_20.1: type = splice_block %.loc24_20.3 [concrete = constants.%facet_type.242] {
// CHECK:STDOUT:       %A.ref.loc24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:       %impl.elem0.loc24_12: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc24_12.1: <bound method> = bound_method %A.ref.loc24, %impl.elem0.loc24_12 [concrete = constants.%Op.bound.6b1]
// CHECK:STDOUT:       %specific_fn.loc24_12: <specific function> = specific_function %impl.elem0.loc24_12, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc24_12.2: <bound method> = bound_method %A.ref.loc24, %specific_fn.loc24_12 [concrete = constants.%bound_method.96c]
// CHECK:STDOUT:       %type.and.loc24_12: init type = call %bound_method.loc24_12.2(%A.ref.loc24, %Empty.ref) [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc24_20: %.d4d = impl_witness_access constants.%impl_witness.3ea, element0 [concrete = constants.%Op.444]
// CHECK:STDOUT:       %bound_method.loc24_20.1: <bound method> = bound_method %type.and.loc24_12, %impl.elem0.loc24_20 [concrete = constants.%Op.bound.b5b]
// CHECK:STDOUT:       %specific_fn.loc24_20: <specific function> = specific_function %impl.elem0.loc24_20, @Op.2(type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:       %bound_method.loc24_20.2: <bound method> = bound_method %type.and.loc24_12, %specific_fn.loc24_20 [concrete = constants.%bound_method.cb6]
// CHECK:STDOUT:       %.loc24_12.1: type = value_of_initializer %type.and.loc24_12 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %.loc24_12.2: type = converted %type.and.loc24_12, %.loc24_12.1 [concrete = constants.%facet_type.d5f]
// CHECK:STDOUT:       %type.and.loc24_20: init type = call %bound_method.loc24_20.2(%.loc24_12.2, %B.ref.loc24) [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc24_20.2: type = value_of_initializer %type.and.loc24_20 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:       %.loc24_20.3: type = converted %type.and.loc24_20, %.loc24_20.2 [concrete = constants.%facet_type.242]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24_6.1: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc24_28.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc24_28.1: type = splice_block %.loc24_28.2 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc24: %facet_type.242 = name_ref T, %T.loc24_6.1 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:       %T.as_type.loc24_28.1: type = facet_access_type %T.ref.loc24 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc24_28.2: type = converted %T.ref.loc24, %T.as_type.loc24_28.1 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc24_28.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.193]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.31d]
// CHECK:STDOUT:   %AA.decl: %AA.type.b97 = fn_decl @AA.1 [concrete = constants.%AA.c3a] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %AA.decl [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .AA = %assoc0
// CHECK:STDOUT:   .BB = <poisoned>
// CHECK:STDOUT:   witness = (%AA.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.783]
// CHECK:STDOUT:   %BB.decl: %BB.type.64d = fn_decl @BB.1 [concrete = constants.%BB.11b] {} {}
// CHECK:STDOUT:   %assoc0: %B.assoc_type = assoc_entity element0, %BB.decl [concrete = constants.%assoc0.019]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BB = %assoc0
// CHECK:STDOUT:   .AA = <poisoned>
// CHECK:STDOUT:   witness = (%BB.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.08d
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.837)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.837) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.43db8b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.630
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d78: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77d: %C.ref as %A.ref {
// CHECK:STDOUT:   %AA.decl: %AA.type.c29 = fn_decl @AA.2 [concrete = constants.%AA.95d] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .AA = %AA.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c94: %C.ref as %B.ref {
// CHECK:STDOUT:   %BB.decl: %BB.type.24d = fn_decl @BB.2 [concrete = constants.%BB.fe8] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .BB = %BB.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.1e6), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.db8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type.f99)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type.f99) = struct_value () [symbolic = %Op (constants.%Op.05a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.51c
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AA.1(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BB.1(@B.%Self: %B.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AA.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BB.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.040: %BitAnd.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.25f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.19f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type.19f)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type.19f)) -> @Op.1.%Self.as_type (%Self.as_type.19f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T.8b3)](%other.param_patt: @Op.2.%T (%T.8b3)) -> @Op.2.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc24_6.1: %facet_type.242) {
// CHECK:STDOUT:   %T.loc24_6.2: %facet_type.242 = bind_symbolic_name T, 0 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:   %T.patt.loc24_6.2: %facet_type.242 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_6.2 (constants.%T.patt.fd9)]
// CHECK:STDOUT:   %T.as_type.loc24_28.2: type = facet_access_type %T.loc24_6.2 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%T.as_type.loc24_28.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.383)]
// CHECK:STDOUT:   %T.as_wit.loc25_4.2: <witness> = facet_access_witness %T.loc24_6.2 [symbolic = %T.as_wit.loc25_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %T.as_type.loc24_28.2, %T.as_wit.loc25_4.2 [symbolic = %A.facet (constants.%A.facet.d6d)]
// CHECK:STDOUT:   %.loc25_4.2: type = fn_type_with_self_type constants.%AA.type.b97, %A.facet [symbolic = %.loc25_4.2 (constants.%.124)]
// CHECK:STDOUT:   %impl.elem0.loc25_4.2: @G.%.loc25_4.2 (%.124) = impl_witness_access %T.as_wit.loc25_4.2, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.2b1)]
// CHECK:STDOUT:   %specific_fn.loc25_4.2: <specific function> = specific_function %impl.elem0.loc25_4.2, @AA.1(%A.facet) [symbolic = %specific_fn.loc25_4.2 (constants.%specific_fn.80f)]
// CHECK:STDOUT:   %B.facet: %B.type = facet_value %T.as_type.loc24_28.2, %T.as_wit.loc25_4.2 [symbolic = %B.facet (constants.%B.facet.1c2)]
// CHECK:STDOUT:   %.loc26_4.2: type = fn_type_with_self_type constants.%BB.type.64d, %B.facet [symbolic = %.loc26_4.2 (constants.%.e3a)]
// CHECK:STDOUT:   %impl.elem0.loc26_4.2: @G.%.loc26_4.2 (%.e3a) = impl_witness_access %T.as_wit.loc25_4.2, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0.f5b)]
// CHECK:STDOUT:   %specific_fn.loc26_4.2: <specific function> = specific_function %impl.elem0.loc26_4.2, @BB.1(%B.facet) [symbolic = %specific_fn.loc26_4.2 (constants.%specific_fn.62f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc24_6.1: %facet_type.242](%t.param_patt: @G.%T.as_type.loc24_28.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref.loc25: @G.%T.as_type.loc24_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %AA.ref.loc25: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc25_4.1: type = converted constants.%T.2df, %T.as_type.loc25 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc25_4.1: <witness> = facet_access_witness constants.%T.2df [symbolic = %T.as_wit.loc25_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc25_4.1: @G.%.loc25_4.2 (%.124) = impl_witness_access %T.as_wit.loc25_4.1, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.2b1)]
// CHECK:STDOUT:     %specific_fn.loc25_4.1: <specific function> = specific_function %impl.elem0.loc25_4.1, @AA.1(constants.%A.facet.d6d) [symbolic = %specific_fn.loc25_4.2 (constants.%specific_fn.80f)]
// CHECK:STDOUT:     %AA.call.loc25: init %empty_tuple.type = call %specific_fn.loc25_4.1()
// CHECK:STDOUT:     %t.ref.loc26: @G.%T.as_type.loc24_28.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %BB.ref.loc26: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type constants.%T.2df [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_4.1: type = converted constants.%T.2df, %T.as_type.loc26 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc26: <witness> = facet_access_witness constants.%T.2df [symbolic = %T.as_wit.loc25_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc26_4.1: @G.%.loc26_4.2 (%.e3a) = impl_witness_access %T.as_wit.loc26, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0.f5b)]
// CHECK:STDOUT:     %specific_fn.loc26_4.1: <specific function> = specific_function %impl.elem0.loc26_4.1, @BB.1(constants.%B.facet.1c2) [symbolic = %specific_fn.loc26_4.2 (constants.%specific_fn.62f)]
// CHECK:STDOUT:     %BB.call.loc26: init %empty_tuple.type = call %specific_fn.loc26_4.1()
// CHECK:STDOUT:     %T.ref.loc28: %facet_type.242 = name_ref T, %T.loc24_6.1 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %AA.ref.loc28: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %T.as_type.loc28: type = facet_access_type %T.ref.loc28 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc28: type = converted %T.ref.loc28, %T.as_type.loc28 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc28: <witness> = facet_access_witness %T.ref.loc28 [symbolic = %T.as_wit.loc25_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc28: @G.%.loc25_4.2 (%.124) = impl_witness_access %T.as_wit.loc28, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.2b1)]
// CHECK:STDOUT:     %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @AA.1(constants.%A.facet.d6d) [symbolic = %specific_fn.loc25_4.2 (constants.%specific_fn.80f)]
// CHECK:STDOUT:     %AA.call.loc28: init %empty_tuple.type = call %specific_fn.loc28()
// CHECK:STDOUT:     %T.ref.loc29: %facet_type.242 = name_ref T, %T.loc24_6.1 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %BB.ref.loc29: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %T.as_type.loc29: type = facet_access_type %T.ref.loc29 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc29: type = converted %T.ref.loc29, %T.as_type.loc29 [symbolic = %T.as_type.loc24_28.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc29: <witness> = facet_access_witness %T.ref.loc29 [symbolic = %T.as_wit.loc25_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc29: @G.%.loc26_4.2 (%.e3a) = impl_witness_access %T.as_wit.loc29, element0 [symbolic = %impl.elem0.loc26_4.2 (constants.%impl.elem0.f5b)]
// CHECK:STDOUT:     %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @BB.1(constants.%B.facet.1c2) [symbolic = %specific_fn.loc26_4.2 (constants.%specific_fn.62f)]
// CHECK:STDOUT:     %BB.call.loc29: init %empty_tuple.type = call %specific_fn.loc29()
// CHECK:STDOUT:     %T.ref.loc41: %facet_type.242 = name_ref T, %T.loc24_6.1 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %A.ref.loc41: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     %AA.ref.loc41: %A.assoc_type = name_ref AA, @A.%assoc0 [concrete = constants.%assoc0.f33]
// CHECK:STDOUT:     %.loc41: %A.type = converted %T.ref.loc41, <error> [concrete = <error>]
// CHECK:STDOUT:     %T.ref.loc52: %facet_type.242 = name_ref T, %T.loc24_6.1 [symbolic = %T.loc24_6.2 (constants.%T.2df)]
// CHECK:STDOUT:     %B.ref.loc52: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %BB.ref.loc52: %B.assoc_type = name_ref BB, @B.%assoc0 [concrete = constants.%assoc0.019]
// CHECK:STDOUT:     %.loc52: %B.type = converted %T.ref.loc52, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.4: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.40a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type.40a)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc67_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc67_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc67_6.3: init %C = class_init (), %.loc67_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc67_6.4: ref %C = temporary %.loc67_6.2, %.loc67_6.3
// CHECK:STDOUT:   %.loc67_8.1: ref %C = converted %.loc67_6.1, %.loc67_6.4
// CHECK:STDOUT:   %facet_value.loc67_12.1: %facet_type.242 = facet_value constants.%C, constants.%impl_witness.1bc [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc67_12.1: %facet_type.242 = converted constants.%C, %facet_value.loc67_12.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc67_12.2: %facet_type.242 = facet_value constants.%C, constants.%impl_witness.1bc [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc67_12.2: %facet_type.242 = converted constants.%C, %facet_value.loc67_12.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%facet_value) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc67_8.2: %C = bind_value %.loc67_8.1
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn(%.loc67_8.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%Self.31d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%Self.783) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.afe) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.3a2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.25f) {
// CHECK:STDOUT:   %Self => constants.%Self.25f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.19f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.8b3
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.db8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:   %T.patt => type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3ea
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.eb8
// CHECK:STDOUT:   %Op => constants.%Op.444
// CHECK:STDOUT:   %require_complete => constants.%complete_type.473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(type) {
// CHECK:STDOUT:   %T => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.2df) {
// CHECK:STDOUT:   %T.loc24_6.2 => constants.%T.2df
// CHECK:STDOUT:   %T.patt.loc24_6.2 => constants.%T.2df
// CHECK:STDOUT:   %T.as_type.loc24_28.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(constants.%A.facet.d6d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(constants.%B.facet.1c2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.40a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%A.type) {
// CHECK:STDOUT:   %Dest => constants.%A.type
// CHECK:STDOUT:   %Dest.patt => constants.%A.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2d2
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.597
// CHECK:STDOUT:   %Convert => constants.%Convert.d1e
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.568
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%B.type) {
// CHECK:STDOUT:   %Dest => constants.%B.type
// CHECK:STDOUT:   %Dest.patt => constants.%B.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.8a4
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.f1c
// CHECK:STDOUT:   %Convert => constants.%Convert.e93
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.f97
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AA.1(@G.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BB.1(@G.%B.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%facet_value) {
// CHECK:STDOUT:   %T.loc24_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.patt.loc24_6.2 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc24_28.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %T.as_wit.loc25_4.2 => constants.%impl_witness.1bc
// CHECK:STDOUT:   %A.facet => constants.%A.facet.beb
// CHECK:STDOUT:   %.loc25_4.2 => constants.%.778
// CHECK:STDOUT:   %impl.elem0.loc25_4.2 => <error>
// CHECK:STDOUT:   %specific_fn.loc25_4.2 => <error>
// CHECK:STDOUT:   %B.facet => constants.%B.facet.5e0
// CHECK:STDOUT:   %.loc26_4.2 => constants.%.555
// CHECK:STDOUT:   %impl.elem0.loc26_4.2 => <error>
// CHECK:STDOUT:   %specific_fn.loc26_4.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

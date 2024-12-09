// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon

// --- extend_impl_generic_interface.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F() -> T;
}

class Param {
  var x: i32;
}

class C {
  extend impl as HasF(Param) {
    fn F() -> Param {
      return {.x = 2};
    }
  }
}

fn G(c: C) {
  let a: i32 = C.F().x;
  var b: i32 = c.F().x;
}

// --- extend_impl_generic_class.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F[self: Self](t: T);
}

class X(U:! type) {
  extend impl as I(U) {
    fn F[self: Self](t: U) { }
  }
}

// CHECK:STDOUT: --- extend_impl_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.1: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.1 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.2: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %HasF.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type.3: type = facet_type <@HasF, @HasF(%Param)> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @HasF(%Param) [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %HasF.type.3, %F.type.2 [template]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.2 = facet_value %C, %C [symbolic]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Param.val: %Param = struct_value (%int_2.2) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.1 = interface_decl @HasF [template = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [template = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.2)]
// CHECK:STDOUT:   %Self.2: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.2), @HasF.%F.type (%F.type.1) [symbolic = %F.assoc_type (constants.%F.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc5_14.2: @HasF.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.1)] {
// CHECK:STDOUT:       %return.patt: @F.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %return.patt: %Param = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Param = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.decl) [template = constants.%interface.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_10.2: type = converted %int.make_type_signed, %.loc9_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8: %Param.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc9_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [template = constants.%HasF.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.1.%HasF.type
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc15_21.1: %struct_type.x.2 = struct_literal (%int_2)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = converted %int_2, %int.convert_checked [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc15_21.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.2 to %.loc15_21.3 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc15_21.5: init %Param = class_init (%.loc15_21.4), %return [template = constants.%Param.val]
// CHECK:STDOUT:   %.loc15_22: init %Param = converted %.loc15_21.1, %.loc15_21.5 [template = constants.%Param.val]
// CHECK:STDOUT:   return %.loc15_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int_32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_signed.loc21, %.loc21_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc21_17: %F.assoc_type.2 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [template = constants.%assoc0.2]
// CHECK:STDOUT:   %F.ref.loc21: %F.assoc_type.2 = name_ref F, %.loc21_17 [template = constants.%assoc0.2]
// CHECK:STDOUT:   %impl.elem0.loc21: %F.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc21_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc21: init %Param = call %impl.elem0.loc21() to %.loc21_20.1
// CHECK:STDOUT:   %.loc21_20.2: ref %Param = temporary %.loc21_20.1, %F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %Param.elem = name_ref x, @Param.%.loc9_8 [template = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc21_21.1: ref %i32 = class_element_access %.loc21_20.2, element0
// CHECK:STDOUT:   %.loc21_21.2: %i32 = bind_value %.loc21_21.1
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc21_21.2
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_10.2: type = converted %int.make_type_signed.loc22, %.loc22_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_17: %F.assoc_type.2 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [template = constants.%assoc0.2]
// CHECK:STDOUT:   %F.ref.loc22: %F.assoc_type.2 = name_ref F, %.loc22_17 [template = constants.%assoc0.2]
// CHECK:STDOUT:   %impl.elem0.loc22: %F.type.2 = interface_witness_access constants.%interface.1, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc22_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc22: init %Param = call %impl.elem0.loc22() to %.loc22_20.1
// CHECK:STDOUT:   %.loc22_20.2: ref %Param = temporary %.loc22_20.1, %F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %Param.elem = name_ref x, @Param.%.loc9_8 [template = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc22_21.1: ref %i32 = class_element_access %.loc22_20.2, element0
// CHECK:STDOUT:   %.loc22_21.2: %i32 = bind_value %.loc22_21.1
// CHECK:STDOUT:   assign %b.var, %.loc22_21.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.2
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [template]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [template]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %I.type.3, %F.type.2 [symbolic]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2, @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.2 = facet_value %X, %X [symbolic]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%F.3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [template = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt.loc8_9.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc8_9.1, runtime_param<invalid> [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc8_9.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %F.assoc_type (constants.%F.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc5_25.2: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %t.patt: @F.1.%T (%T) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.1.%T (%T) = value_param_pattern %t.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_14.1: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc5_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.1.%T (%T) = value_param runtime_param1
// CHECK:STDOUT:       %t: @F.1.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc5_25.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.2: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%U) [symbolic = %F.type (constants.%F.type.3)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.3) = struct_value () [symbolic = %F (constants.%F.3)]
// CHECK:STDOUT:   %interface.loc9_23.2: <witness> = interface_witness (%F) [symbolic = %interface.loc9_23.2 (constants.%interface)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.3) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.3)] {
// CHECK:STDOUT:       %self.patt: @F.2.%X (%X) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%X (%X) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %t.patt: @F.2.%U (%U) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.2.%U (%U) = value_param_pattern %t.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc10 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.2.%X (%X) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%X (%X) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.2.%U (%U) = value_param runtime_param1
// CHECK:STDOUT:       %t: @F.2.%U (%U) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %interface.loc9_23.1: <witness> = interface_witness (%F.decl) [symbolic = %interface.loc9_23.2 (constants.%interface)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %interface.loc9_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U.loc8_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_9.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.2)> [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl [template] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.1: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: type = specific_constant @impl.%I.type.loc9_21.1, @impl(constants.%U) [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X
// CHECK:STDOUT:     extend %.loc9
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%t.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%X (%X)](%t.param_patt: @F.2.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@impl.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.2 => constants.%I.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %interface.loc9_23.2 => constants.%interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@F.2.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.2 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@X.%U.loc8_9.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:

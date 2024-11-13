// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon

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

// CHECK:STDOUT: --- extend_impl_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.1: type = generic_interface_type @HasF [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %HasF: %HasF.type.1 = struct_value () [template]
// CHECK:STDOUT:   %HasF.type.2: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %HasF.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Param, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %HasF.type.3: type = facet_type <@HasF, @HasF(%Param)> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @HasF(%Param) [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %HasF.type.3, %F.type.2 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%F.3) [template]
// CHECK:STDOUT:   %.9: type = struct_type {} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.12: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %Param = struct_value (%.12) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.1 = interface_decl @HasF [template = constants.%HasF] {
// CHECK:STDOUT:     %T.patt.loc11_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_16.1, runtime_param<invalid> [symbolic = %T.patt.loc11_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [template = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc11_16.1: type) {
// CHECK:STDOUT:   %T.loc11_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc11_16.2)> [symbolic = %HasF.type (constants.%HasF.type.2)]
// CHECK:STDOUT:   %Self.2: %HasF.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc11_16.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc12_14.2: type = assoc_entity_type @HasF.%HasF.type (%HasF.type.2), @HasF.%F.type (%F.type.1) [symbolic = %.loc12_14.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc12_14.3: @HasF.%.loc12_14.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc12_14.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.1)] {
// CHECK:STDOUT:       %return.patt: @F.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%T (%T) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc11_16.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_14.1: @HasF.%.loc12_14.2 (%.1) = assoc_entity element0, %F.decl [symbolic = %.loc12_14.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc12_14.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %return.patt: %Param = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Param = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = interface_witness (%F.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_32, %.loc16_10.1 [template = i32]
// CHECK:STDOUT:   %.loc16_8: %.3 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.1 = name_ref HasF, file.%HasF.decl [template = constants.%HasF]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [template = constants.%HasF.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   extend @impl.%HasF.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc11_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_20: i32 = int_value 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_21.1: %.4 = struct_literal (%.loc22_20)
// CHECK:STDOUT:   %.loc22_21.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc22_21.3: init i32 = initialize_from %.loc22_20 to %.loc22_21.2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_21.4: init %Param = class_init (%.loc22_21.3), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc22_22: init %Param = converted %.loc22_21.1, %.loc22_21.4 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc22_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc28_10.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:   %.loc28_10.2: type = converted %int.make_type_32.loc28, %.loc28_10.1 [template = i32]
// CHECK:STDOUT:   %C.ref.loc28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc28_17.1: %.6 = specific_constant @HasF.%.loc12_14.1, @HasF(constants.%Param) [template = constants.%.7]
// CHECK:STDOUT:   %F.ref.loc28: %.6 = name_ref F, %.loc28_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc28_17.2: %F.type.2 = interface_witness_access constants.%.8, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc28_19.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc28: init %Param = call %.loc28_17.2() to %.loc28_19.1
// CHECK:STDOUT:   %.loc28_19.2: ref %Param = temporary %.loc28_19.1, %F.call.loc28
// CHECK:STDOUT:   %x.ref.loc28: %.3 = name_ref x, @Param.%.loc16_8 [template = @Param.%.loc16_8]
// CHECK:STDOUT:   %.loc28_21.1: ref i32 = class_element_access %.loc28_19.2, element0
// CHECK:STDOUT:   %.loc28_21.2: i32 = bind_value %.loc28_21.1
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc28_21.2
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_10.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_10.2: type = converted %int.make_type_32.loc29, %.loc29_10.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc29_17.1: %.6 = specific_constant @HasF.%.loc12_14.1, @HasF(constants.%Param) [template = constants.%.7]
// CHECK:STDOUT:   %F.ref.loc29: %.6 = name_ref F, %.loc29_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc29_17.2: %F.type.2 = interface_witness_access constants.%.8, element0 [template = constants.%F.3]
// CHECK:STDOUT:   %.loc29_19.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc29: init %Param = call %.loc29_17.2() to %.loc29_19.1
// CHECK:STDOUT:   %.loc29_19.2: ref %Param = temporary %.loc29_19.1, %F.call.loc29
// CHECK:STDOUT:   %x.ref.loc29: %.3 = name_ref x, @Param.%.loc16_8 [template = @Param.%.loc16_8]
// CHECK:STDOUT:   %.loc29_21.1: ref i32 = class_element_access %.loc29_19.2, element0
// CHECK:STDOUT:   %.loc29_21.2: i32 = bind_value %.loc29_21.1
// CHECK:STDOUT:   assign %b.var, %.loc29_21.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc11_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(@HasF.%T.loc11_16.2) {
// CHECK:STDOUT:   %T.loc11_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc11_16.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc11_16.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc12_14.2 => constants.%.6
// CHECK:STDOUT:   %.loc12_14.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Param, constants.%C) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT: }
// CHECK:STDOUT:

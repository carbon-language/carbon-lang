// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/self_in_class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/self_in_class.carbon

interface DefaultConstructible {
  fn Make() -> Self;
}

class C {}

class A {
  impl C as DefaultConstructible {
    // `Self` here refers to `C`, not `A`.
    // TODO: Revisit this once #3714 is resolved.
    fn Make() -> Self { return {}; }
  }
}

// CHECK:STDOUT: --- self_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DefaultConstructible.type: type = facet_type <@DefaultConstructible> [template]
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1 [template]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Make.assoc_type: type = assoc_entity_type %DefaultConstructible.type, %Make.type.1 [template]
// CHECK:STDOUT:   %assoc0: %Make.assoc_type = assoc_entity element0, @DefaultConstructible.%Make.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %DefaultConstructible.facet: %DefaultConstructible.type = facet_value %C, %C [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Make.2) [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .DefaultConstructible = %DefaultConstructible.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructible.decl: type = interface_decl @DefaultConstructible [template = constants.%DefaultConstructible.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultConstructible {
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Make.decl: %Make.type.1 = fn_decl @Make.1 [template = constants.%Make.1] {
// CHECK:STDOUT:     %return.patt: @Make.1.%Self.as_type.loc12_16.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Make.1.%Self.as_type.loc12_16.1 (%Self.as_type) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %DefaultConstructible.type = name_ref Self, @DefaultConstructible.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc12_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc12: type = converted %Self.ref, %Self.as_type.loc12_16.2 [symbolic = %Self.as_type.loc12_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @Make.1.%Self.as_type.loc12_16.1 (%Self.as_type) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Make.1.%Self.as_type.loc12_16.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Make.assoc_type = assoc_entity element0, %Make.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Make = %assoc0
// CHECK:STDOUT:   witness = (%Make.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C.ref as %DefaultConstructible.ref {
// CHECK:STDOUT:   %Make.decl: %Make.type.2 = fn_decl @Make.2 [template = constants.%Make.2] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C.ref [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Make.decl) [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %DefaultConstructible.ref: type = name_ref DefaultConstructible, file.%DefaultConstructible.decl [template = constants.%DefaultConstructible.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@DefaultConstructible.%Self: %DefaultConstructible.type) {
// CHECK:STDOUT:   %Self: %DefaultConstructible.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc12_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%Self.as_type.loc12_16.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc21_33.2: init %C = class_init (), %return [template = constants.%C.val]
// CHECK:STDOUT:   %.loc21_34: init %C = converted %.loc21_33.1, %.loc21_33.2 [template = constants.%C.val]
// CHECK:STDOUT:   return %.loc21_34 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc12_16.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%DefaultConstructible.facet) {
// CHECK:STDOUT:   %Self => constants.%DefaultConstructible.facet
// CHECK:STDOUT:   %Self.as_type.loc12_16.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

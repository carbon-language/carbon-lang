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
    // `Self` here refers to `i32`, not `A`.
    // TODO: Revisit this once #3714 is resolved.
    fn Make() -> Self { return {}; }
  }
}

// CHECK:STDOUT: --- self_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @DefaultConstructible [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Make.type.1: type = fn_type @Make.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Make.1: %Make.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %Make.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @DefaultConstructible.%Make.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Make.type.2: type = fn_type @Make.2 [template]
// CHECK:STDOUT:   %Make.2: %Make.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%Make.2) [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .DefaultConstructible = %DefaultConstructible.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructible.decl: type = interface_decl @DefaultConstructible [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultConstructible {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Make.decl: %Make.type.1 = fn_decl @Make.1 [template = constants.%Make.1] {
// CHECK:STDOUT:     %Self.ref: %.1 = name_ref Self, %Self [symbolic = @Make.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_16.1: type = facet_type_access %Self.ref [symbolic = @Make.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_16.2: type = converted %Self.ref, %.loc12_16.1 [symbolic = @Make.1.%Self (constants.%Self)]
// CHECK:STDOUT:     %return.var: ref @Make.1.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_20: %.3 = assoc_entity element0, %Make.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Make = %.loc12_20
// CHECK:STDOUT:   witness = (%Make.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.1 {
// CHECK:STDOUT:   %Make.decl: %Make.type.2 = fn_decl @Make.2 [template = constants.%Make.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Make.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %DefaultConstructible.ref: type = name_ref DefaultConstructible, file.%DefaultConstructible.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make.1(@DefaultConstructible.%Self: %.1) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.1.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> @impl.%return.var: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_33.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc21_33.2: init %C = class_init (), @impl.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_34: init %C = converted %.loc21_33.1, %.loc21_33.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc21_34 to @impl.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

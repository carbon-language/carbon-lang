// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface DefaultConstructible {
  fn Make() -> Self;
}

class A {
  impl i32 as DefaultConstructible {
    // `Self` here refers to `i32`, not `A`.
    // TODO: Revisit this once #3714 is resolved.
    fn Make() -> Self { return 0; }
  }
}

// CHECK:STDOUT: --- self_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @DefaultConstructible [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @DefaultConstructible, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in DefaultConstructible> = assoc_entity element0, @DefaultConstructible.%Make [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (@impl.%Make) [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .DefaultConstructible = %DefaultConstructible.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultConstructible.decl: type = interface_decl @DefaultConstructible [template = constants.%.1] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @DefaultConstructible {
// CHECK:STDOUT:   %Self: DefaultConstructible = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make.1 [template] {
// CHECK:STDOUT:     %Self.ref: DefaultConstructible = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_16.1: type = facet_type_access %Self.ref [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_16.2: type = converted %Self.ref, %.loc8_16.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_20: <associated <function> in DefaultConstructible> = assoc_entity element0, %Make [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Make = %.loc8_20
// CHECK:STDOUT:   witness = (%Make)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as DefaultConstructible {
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make.2 [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, i32 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Make) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Make = %Make
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %DefaultConstructible.ref: type = name_ref DefaultConstructible, file.%DefaultConstructible.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.1() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_derived_to_base.carbon

base class A1 {
  var a: i32;
}

base class A2 {
  var a: i32;
}

class B2 {
  extend base: A2;
  var b: i32;
}

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:38: ERROR: Cannot implicitly convert from `B2*` to `A1*`.
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: Type `B2*` does not implement interface `ImplicitAs`.
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+6]]:47: ERROR: Cannot implicitly convert from `Incomplete*` to `A2*`.
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:47: Type `Incomplete*` does not implement interface `ImplicitAs`.
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A1, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %A2, i32 [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B2, %A2 [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %B2, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %A2, .b: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %B2 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %A1 [template]
// CHECK:STDOUT:   %ConvertUnrelated.type: type = fn_type @ConvertUnrelated [template]
// CHECK:STDOUT:   %ConvertUnrelated: %ConvertUnrelated.type = struct_value () [template]
// CHECK:STDOUT:   %.11: type = struct_type {.base: %.5, .b: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.14: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.14) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.14 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %.14, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.17: type = interface_type @ImplicitAs, @ImplicitAs(%.10) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.10) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %.17, %Convert.type.2 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.20: %.15 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.21: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %.22: type = ptr_type %A2 [template]
// CHECK:STDOUT:   %ConvertIncomplete.type: type = fn_type @ConvertIncomplete [template]
// CHECK:STDOUT:   %ConvertIncomplete: %ConvertIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %.23: type = interface_type @ImplicitAs, @ImplicitAs(%.22) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert, @ImplicitAs(%.22) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.24: type = assoc_entity_type %.23, %Convert.type.3 [template]
// CHECK:STDOUT:   %.25: %.24 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.2 (%.15) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.20)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .ConvertUnrelated = %ConvertUnrelated.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .ConvertIncomplete = %ConvertIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [template = constants.%A1] {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [template = constants.%A2] {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [template = constants.%B2] {}
// CHECK:STDOUT:   %ConvertUnrelated.decl: %ConvertUnrelated.type = fn_decl @ConvertUnrelated [template = constants.%ConvertUnrelated] {
// CHECK:STDOUT:     %B2.ref: type = name_ref B2, %B2.decl [template = constants.%B2]
// CHECK:STDOUT:     %.loc31_26: type = ptr_type %B2 [template = constants.%.9]
// CHECK:STDOUT:     %p.loc31_21.1: %.9 = param p
// CHECK:STDOUT:     @ConvertUnrelated.%p: %.9 = bind_name p, %p.loc31_21.1
// CHECK:STDOUT:     %A1.ref: type = name_ref A1, %A1.decl [template = constants.%A1]
// CHECK:STDOUT:     %.loc31_34: type = ptr_type %A1 [template = constants.%.10]
// CHECK:STDOUT:     @ConvertUnrelated.%return: ref %.10 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %ConvertIncomplete.decl: %ConvertIncomplete.type = fn_decl @ConvertIncomplete [template = constants.%ConvertIncomplete] {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc41_35: type = ptr_type %Incomplete [template = constants.%.21]
// CHECK:STDOUT:     %p.loc41_22.1: %.21 = param p
// CHECK:STDOUT:     @ConvertIncomplete.%p: %.21 = bind_name p, %p.loc41_22.1
// CHECK:STDOUT:     %A2.ref: type = name_ref A2, %A2.decl [template = constants.%A2]
// CHECK:STDOUT:     %.loc41_43: type = ptr_type %A2 [template = constants.%.22]
// CHECK:STDOUT:     @ConvertIncomplete.%return: ref %.22 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %Self: %.14 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.14), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.15) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.3 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_32, %.loc16_10.1 [template = i32]
// CHECK:STDOUT:   %.loc16_8: %.4 = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .a = %.loc16_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:   %.loc20: %.6 = base_decl %A2, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc21_10.2: type = converted %int.make_type_32, %.loc21_10.1 [template = i32]
// CHECK:STDOUT:   %.loc21_8: %.7 = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .b = %.loc21_8
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertUnrelated(%p: %.9) -> %.10 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_46.1: init type = call constants.%ImplicitAs(constants.%.10) [template = constants.%.17]
// CHECK:STDOUT:   %.loc31_46.2: %.18 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.10) [template = constants.%.19]
// CHECK:STDOUT:   %Convert.ref: %.18 = name_ref Convert, %.loc31_46.2 [template = constants.%.19]
// CHECK:STDOUT:   %.loc31_46.3: %.10 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.14)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %Self: %.14 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertIncomplete(%p: %.21) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.21 = name_ref p, %p
// CHECK:STDOUT:   %.loc41_55.1: init type = call constants.%ImplicitAs(constants.%.22) [template = constants.%.23]
// CHECK:STDOUT:   %.loc41_55.2: %.24 = specific_constant imports.%import_ref.4, @ImplicitAs(constants.%.22) [template = constants.%.25]
// CHECK:STDOUT:   %Convert.ref: %.24 = name_ref Convert, %.loc41_55.2 [template = constants.%.25]
// CHECK:STDOUT:   %.loc41_55.3: %.22 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.10) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.17
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.18
// CHECK:STDOUT:   %.3 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.22) {
// CHECK:STDOUT:   %Dest => constants.%.22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.23
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.2 => constants.%.24
// CHECK:STDOUT:   %.3 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:

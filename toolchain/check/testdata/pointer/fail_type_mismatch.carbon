// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon

fn ConstMismatch(p: const {}*) -> const ({}*) {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `const {}*` to `const ({}*)`
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:3: note: type `const {}*` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = const_type %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = const_type %.4 [template]
// CHECK:STDOUT:   %ConstMismatch.type: type = fn_type @ConstMismatch [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %ConstMismatch: %ConstMismatch.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%.5) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%.5) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.11: %.7 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.1 (%.7) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ConstMismatch = %ConstMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ConstMismatch.decl: %ConstMismatch.type = fn_decl @ConstMismatch [template = constants.%ConstMismatch] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_28: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_21.1: type = converted %.loc11_28, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_21.2: type = const_type %.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_29: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %p.param: %.3 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:     %.loc11_43: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_44.1: type = converted %.loc11_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_44.2: type = ptr_type %.1 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_35: type = const_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %return: ref %.5 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.7) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p: %.3) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%.5) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc18_11.1: %.9 = specific_constant imports.%import_ref.3, @ImplicitAs(constants.%.5) [template = constants.%.10]
// CHECK:STDOUT:   %Convert.ref: %.9 = name_ref Convert, %.loc18_11.1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_11.2: %.5 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
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
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.5) {
// CHECK:STDOUT:   %Dest => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.9
// CHECK:STDOUT:   %.2 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:

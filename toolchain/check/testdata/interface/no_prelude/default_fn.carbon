// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/default_fn.carbon

class C {
  interface I {
    // TODO: Use `default` here.
    fn F() {
      // I and F are both complete here, and the impl below is in scope.
      var c: C = {};
      c.(I.F)();
    }
  }

  impl C as I {
    fn F() {}
  }
}

// CHECK:STDOUT: --- default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I[%Self] [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.6: type = struct_type {} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I[%C] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I[%Self.2] [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.1, @I.%F.type (%F.type.1) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @I.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = %F (constants.%F.1)] {}
// CHECK:STDOUT:   %.loc14: @I.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .F = %.loc14
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %.1 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self.1: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I[%Self] [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, @F.1.%F.type (%F.type.1) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %F.decl: @F.1.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = %F.decl (constants.%F.1)] {}
// CHECK:STDOUT:   %.3: @F.1.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.var: ref %C = var c
// CHECK:STDOUT:     %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:     %.loc16_19.1: %.6 = struct_literal ()
// CHECK:STDOUT:     %.loc16_19.2: init %C = class_init (), %c.var [template = constants.%struct]
// CHECK:STDOUT:     %.loc16_20: init %C = converted %.loc16_19.1, %.loc16_19.2 [template = constants.%struct]
// CHECK:STDOUT:     assign %c.var, %.loc16_20
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %I.ref: type = name_ref I, @C.%I.decl [template = constants.%.1]
// CHECK:STDOUT:     %F.ref: @F.1.%.2 (%.3) = name_ref F, @I.%.loc14 [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:     %.1: %F.type.3 = interface_witness_access @impl.%.1, element0 [template = constants.%F.2]
// CHECK:STDOUT:     %F.call: init %.2 = call %.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[constants.%Self] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[@I.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[constants.%C] {
// CHECK:STDOUT:   %Self.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[@F.1.%Self] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

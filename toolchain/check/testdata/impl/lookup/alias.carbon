// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/alias.carbon

interface HasF {
  fn F();
}

class C {
  alias G = HasF.F;
}

impl C as HasF {
  fn F() {}
}

fn G(c: C) {
  C.G();
  c.G();
}

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @HasF [template]
// CHECK:STDOUT:   %Self: HasF = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @HasF, <type of F> [template]
// CHECK:STDOUT:   %.4: <associated <type of F> in HasF> = assoc_entity element0, @HasF.%F.decl [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: <type of G> = struct_value () [template]
// CHECK:STDOUT:   %.7: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %HasF.decl: type = interface_decl @HasF [template = constants.%.1] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, %HasF.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: <type of G> = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc23_6.1: C = param c
// CHECK:STDOUT:     @G.%c: C = bind_name c, %c.loc23_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @HasF {
// CHECK:STDOUT:   %Self: HasF = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F.1 [template = constants.%F.1] {}
// CHECK:STDOUT:   %.loc12: <associated <type of F> in HasF> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as HasF {
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F.2 [template = constants.%F.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%F.decl) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %HasF.ref: type = name_ref HasF, file.%HasF.decl [template = constants.%.1]
// CHECK:STDOUT:   %F.ref: <associated <type of F> in HasF> = name_ref F, @HasF.%.loc12 [template = constants.%.4]
// CHECK:STDOUT:   %G: <associated <type of F> in HasF> = bind_alias G, @HasF.%.loc12 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %G.ref.loc24: <associated <type of F> in HasF> = name_ref G, @C.%G [template = constants.%.4]
// CHECK:STDOUT:   %.1: <type of F> = interface_witness_access @impl.%.1, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call.loc24: init () = call %.1()
// CHECK:STDOUT:   %c.ref: C = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc25: <associated <type of F> in HasF> = name_ref G, @C.%G [template = constants.%.4]
// CHECK:STDOUT:   %.2: <type of F> = interface_witness_access @impl.%.1, element0 [template = constants.%F.2]
// CHECK:STDOUT:   %F.call.loc25: init () = call %.2()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

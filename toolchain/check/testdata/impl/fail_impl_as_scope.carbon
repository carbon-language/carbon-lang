// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_impl_as_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_impl_as_scope.carbon

// --- fail_global.carbon

library "[[@TEST_NAME]]";

interface Simple {
  fn F();
}

// CHECK:STDERR: fail_global.carbon:[[@LINE+4]]:6: error: `impl as` can only be used in a class [ImplAsOutsideClass]
// CHECK:STDERR: impl as Simple {
// CHECK:STDERR:      ^~
// CHECK:STDERR:
impl as Simple {
  fn F() {}
}

// --- fail_function.carbon

library "[[@TEST_NAME]]";

interface Simple {
  fn F();
}

fn Function() {
  // CHECK:STDERR: fail_function.carbon:[[@LINE+4]]:8: error: `impl as` can only be used in a class [ImplAsOutsideClass]
  // CHECK:STDERR:   impl as Simple {
  // CHECK:STDERR:        ^~
  // CHECK:STDERR:
  impl as Simple {
    fn F() {}
  }
}

// CHECK:STDOUT: --- fail_global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [template]
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.e2e: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.df8: %F.type.e2e = struct_value () [template]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type %Simple.type [template]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.%F.decl [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.fe5: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.32d: %F.type.fe5 = struct_value () [template]
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
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%Simple.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.e2e = fn_decl @F.1 [template = constants.%F.df8] {} {}
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %Simple.ref {
// CHECK:STDOUT:   %F.decl: %F.type.fe5 = fn_decl @F.2 [template = constants.%F.32d] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [template]
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.e2e: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.df8: %F.type.e2e = struct_value () [template]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type %Simple.type [template]
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, @Simple.%F.decl [template]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.bc7: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.0a2: %F.type.bc7 = struct_value () [template]
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
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [template = constants.%Simple.type] {} {}
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.e2e = fn_decl @F.1 [template = constants.%F.df8] {} {}
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %Simple.ref {
// CHECK:STDOUT:   %F.decl: %F.type.bc7 = fn_decl @F.2 [template = constants.%F.0a2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @Function.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [template = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(<error>) {}
// CHECK:STDOUT:

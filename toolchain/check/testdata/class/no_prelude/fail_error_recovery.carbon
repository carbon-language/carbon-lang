// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/fail_error_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/fail_error_recovery.carbon

// --- fail_virtual_fn_in_invalid_context.carbon

// CHECK:STDERR: fail_virtual_fn_in_invalid_context.carbon:[[@LINE+4]]:10: error: name `error_not_found` not found [NameNotFound]
// CHECK:STDERR: fn F(N:! error_not_found) {
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(N:! error_not_found) {
  base class C {
    virtual fn Foo[self: Self]() {}
  }

  base class D {
    extend base: C;
  }
}

// CHECK:STDOUT: --- fail_virtual_fn_in_invalid_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %C} [concrete]
// CHECK:STDOUT:   %complete_type.1d0: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .error_not_found = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: <error> = symbolic_binding_pattern N, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %error_not_found.ref: <error> = name_ref error_not_found, <error> [concrete = <error>]
// CHECK:STDOUT:     %N: <error> = bind_symbolic_name N, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@F.%N: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Foo.decl: <error> = fn_decl @Foo [concrete = <error>] {
// CHECK:STDOUT:       %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: <error> = value_param_pattern %self.patt, call_param0 [concrete = <error>]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: <error> = value_param call_param0
// CHECK:STDOUT:       %Self.ref: <error> = name_ref Self, <error> [concrete = <error>]
// CHECK:STDOUT:       %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: <vtable> = vtable (%Foo.decl) [concrete = <error>]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = <error>
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(@F.%N: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.ref: type = name_ref C, @F.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc12: <error> = base_decl %C.ref, element0 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.1d0]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = <error>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .base = %.loc12
// CHECK:STDOUT:     extend %C.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:     %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Foo(@F.%N: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn[%self.param_patt: <error>]() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(<error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Foo(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(<error>) {}
// CHECK:STDOUT:

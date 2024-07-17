// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_generic_default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_todo_generic_default_fn.carbon

interface I(T:! type) {
  // TODO: Use `default` here.
  fn F[self: Self]() -> Self;
}

// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE-6]]:3: Name is previously declared here.
// CHECK:STDERR:   fn F[self: Self]() -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
fn I(T:! type).F[self: Self]() -> Self { return self; }

// CHECK:STDOUT: --- fail_todo_generic_default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @I, file.%I.decl(%T) [symbolic]
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @I, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %Self.2: <unexpected>.inst+28 (%.2) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.5: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.5] {
// CHECK:STDOUT:     %T.loc22_6.1: type = param T
// CHECK:STDOUT:     %T.loc22_6.2: type = bind_symbolic_name T 0, %T.loc22_6.1 [symbolic = %T.loc22_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc22_24.1: <unexpected>.inst+44 (%.2) = specific_constant @I.%Self, %I.decl(constants.%T) [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %Self.ref.loc22_24: <unexpected>.inst+44 (%.2) = name_ref Self, %.loc22_24.1 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %.loc22_24.2: type = facet_type_access %Self.ref.loc22_24 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %.loc22_24.3: type = converted %Self.ref.loc22_24, %.loc22_24.2 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %self.loc22_18.1: file.%.loc22_24.1 (%Self.2) = param self
// CHECK:STDOUT:     @.1.%self: file.%.loc22_24.1 (%Self.2) = bind_name self, %self.loc22_18.1
// CHECK:STDOUT:     %.loc22_35.1: <unexpected>.inst+44 (%.2) = specific_constant @I.%Self, %I.decl(constants.%T) [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %Self.ref.loc22_35: <unexpected>.inst+44 (%.2) = name_ref Self, %.loc22_35.1 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %.loc22_35.2: type = facet_type_access %Self.ref.loc22_35 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     %.loc22_35.3: type = converted %Self.ref.loc22_35, %.loc22_35.2 [symbolic = %.loc22_24.1 (constants.%Self.2)]
// CHECK:STDOUT:     @.1.%return: ref %Self.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %Self: <unexpected>.inst+28 (%.2) = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc13_14.1: %.2 = specific_constant %Self, file.%I.decl(constants.%T) [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc13_14: %.2 = name_ref Self, %.loc13_14.1 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %.loc13_14.2: type = facet_type_access %Self.ref.loc13_14 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %.loc13_14.3: type = converted %Self.ref.loc13_14, %.loc13_14.2 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %self.loc13_8.1: @I.%.loc13_14.1 (%Self.1) = param self
// CHECK:STDOUT:     %self.loc13_8.2: @I.%.loc13_14.1 (%Self.1) = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %.loc13_25.1: %.2 = specific_constant %Self, file.%I.decl(constants.%T) [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc13_25: %.2 = name_ref Self, %.loc13_25.1 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %.loc13_25.2: type = facet_type_access %Self.ref.loc13_25 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %.loc13_25.3: type = converted %Self.ref.loc13_25, %.loc13_25.2 [symbolic = %.loc13_14.1 (constants.%Self.1)]
// CHECK:STDOUT:     %return.var: ref %Self.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_29: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc13_29
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@I.%self.loc13_8.2: @I.%.loc13_14.1 (%Self.1)]() -> %Self.1
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type, @I.%Self: <unexpected>.inst+28 (%.2)];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1[%self: file.%.loc22_24.1 (%Self.2)]() -> %Self.2
// CHECK:STDOUT:     generic [file.%T.loc22_6.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: file.%.loc22_24.1 (%Self.2) = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%I.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: definition:
// CHECK:STDOUT:   <unexpected>.inst+28 => constants.%.2
// CHECK:STDOUT:   @I.%Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.%F.decl(constants.%T, constants.%Self.1) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @I.%.loc13_14.1 => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%I.decl(file.%T.loc11_13.2) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%I.decl(file.%T.loc22_6.2) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc22_6.2 => constants.%T
// CHECK:STDOUT:   <unexpected>.inst+44 => constants.%.2
// CHECK:STDOUT:   file.%.loc22_24.1 => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

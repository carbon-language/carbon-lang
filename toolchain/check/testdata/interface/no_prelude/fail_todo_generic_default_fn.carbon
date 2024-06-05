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
  // CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+8]]:14: ERROR: Cannot implicitly convert from `<cannot stringify inst+7>` to `type`.
  // CHECK:STDERR:   fn F[self: Self]() -> Self;
  // CHECK:STDERR:              ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+4]]:25: ERROR: Cannot implicitly convert from `<cannot stringify inst+7>` to `type`.
  // CHECK:STDERR:   fn F[self: Self]() -> Self;
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR:
  fn F[self: Self]() -> Self;
}

// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+14]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE-6]]:3: Name is previously declared here.
// CHECK:STDERR:   fn F[self: Self]() -> Self;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+7]]:24: ERROR: Cannot implicitly convert from `<cannot stringify inst+7>` to `type`.
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_generic_default_fn.carbon:[[@LINE+3]]:35: ERROR: Cannot implicitly convert from `<cannot stringify inst+7>` to `type`.
// CHECK:STDERR: fn I(T:! type).F[self: Self]() -> Self { return self; }
// CHECK:STDERR:                                   ^~~~
fn I(T:! type).F[self: Self]() -> Self { return self; }

// CHECK:STDOUT: --- fail_todo_generic_default_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: <type of I> = struct_value () [template]
// CHECK:STDOUT:   %Self: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <type of F> [template]
// CHECK:STDOUT:   %.3: <associated <type of F> in I> = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: <type of I> = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <type of <invalid>> = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %T.loc38_6.1: type = param T
// CHECK:STDOUT:     %T.loc38_6.2: type = bind_symbolic_name T 0, %T.loc38_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %Self.ref.loc38_24: <cannot stringify inst+7> = name_ref Self, @I.%Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc38_18.1: <error> = param self
// CHECK:STDOUT:     @.1.%self: <error> = bind_name self, %self.loc38_18.1
// CHECK:STDOUT:     %Self.ref.loc38_35: <cannot stringify inst+7> = name_ref Self, @I.%Self [symbolic = constants.%Self]
// CHECK:STDOUT:     @.1.%return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: <cannot stringify inst+7> = bind_symbolic_name Self 1 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc21_14: <cannot stringify inst+7> = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc21_8.1: <error> = param self
// CHECK:STDOUT:     %self.loc21_8.2: <error> = bind_name self, %self.loc21_8.1
// CHECK:STDOUT:     %Self.ref.loc21_25: <cannot stringify inst+7> = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: <associated <type of F> in I> = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc21
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@I.%self.loc21_8.2: <error>]() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1[%self: <error>]() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: <error> = name_ref self, %self
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

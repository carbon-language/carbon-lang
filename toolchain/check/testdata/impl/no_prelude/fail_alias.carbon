// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/fail_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/fail_alias.carbon

interface I {}
class C {}

alias AI = I;
alias AC = C;

impl AC as AI {}

// CHECK:STDERR: fail_alias.carbon:[[@LINE+6]]:1: error: redefinition of `impl AC as AI`
// CHECK:STDERR: impl AC as AI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_alias.carbon:[[@LINE-5]]:1: note: previous definition was here
// CHECK:STDERR: impl AC as AI {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
impl AC as AI {}

// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .AI = %AI
// CHECK:STDOUT:     .AC = %AC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %AI: type = bind_alias AI, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %AC: type = bind_alias AC, %C.decl [template = constants.%C]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %AC.ref.loc17: type = name_ref AC, file.%AC [template = constants.%C]
// CHECK:STDOUT:     %AI.ref.loc17: type = name_ref AI, file.%AI [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %AC.ref.loc25: type = name_ref AC, file.%AC [template = constants.%C]
// CHECK:STDOUT:     %AI.ref.loc25: type = name_ref AI, file.%AI [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %AC.ref.loc17 as %AI.ref.loc17 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <unexpected>.inst+18.loc17_15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

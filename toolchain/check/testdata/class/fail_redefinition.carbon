// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_redefinition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_redefinition.carbon

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: error: redefinition of `class Class` [RedeclRedef]
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

fn Class.F() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+4]]:10: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn Class.G() {}
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: error: redefinition of `fn I` [RedeclRedef]
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-26]]:3: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR:   fn I() {}
// CHECK:STDERR:   ^~~~~~~~
fn Class.I() {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %H.type.ba6: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %H.9d5: %H.type.ba6 = struct_value () [template]
// CHECK:STDOUT:   %I.type.92e: type = fn_type @I.1 [template]
// CHECK:STDOUT:   %I.f31: %I.type.92e = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %.a95: type = class_type @.1 [template]
// CHECK:STDOUT:   %G.type.aab: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.c83: %G.type.aab = struct_value () [template]
// CHECK:STDOUT:   %H.type.607: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %H.f71: %H.type.607 = struct_value () [template]
// CHECK:STDOUT:   %I.type.406: type = fn_type @I.2 [template]
// CHECK:STDOUT:   %I.f46: %I.type.406 = struct_value () [template]
// CHECK:STDOUT:   %G.type.bf5: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.5fa: %G.type.bf5 = struct_value () [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.a95] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type.bf5 = fn_decl @G.2 [template = constants.%G.5fa] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.ba6 = fn_decl @H.1 [template = constants.%H.9d5] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.92e = fn_decl @I.1 [template = constants.%I.f31] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.ba6 = fn_decl @H.1 [template = constants.%H.9d5] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.92e = fn_decl @I.1 [template = constants.%I.f31] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .G = file.%G.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %G.decl: %G.type.aab = fn_decl @G.1 [template = constants.%G.c83] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.607 = fn_decl @H.2 [template = constants.%H.f71] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.406 = fn_decl @I.2 [template = constants.%I.f46] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.a95
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: error: redefinition of `fn I` [RedeclRedef]
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-26]]:3: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR:   fn I() {}
// CHECK:STDERR:   ^~~~~~~~
// CHECK:STDERR:
fn Class.I() {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.91d: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %H.d38: %H.type.91d = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.2b6: type = fn_type @I.1 [concrete]
// CHECK:STDOUT:   %I.c9a: %I.type.2b6 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.a95: type = class_type @.2 [concrete]
// CHECK:STDOUT:   %G.type.bf6: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.e39: %G.type.bf6 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.e2f: type = fn_type @H.2 [concrete]
// CHECK:STDOUT:   %H.382: %H.type.e2f = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.b27: type = fn_type @I.2 [concrete]
// CHECK:STDOUT:   %I.a7f: %I.type.b27 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.621: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.f0c: %G.type.621 = struct_value () [concrete]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [concrete]
// CHECK:STDOUT:   %.d22: %.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %.decl.loc24: type = class_decl @.2 [concrete = constants.%.a95] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type.621 = fn_decl @G.2 [concrete = constants.%G.f0c] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.91d = fn_decl @H.1 [concrete = constants.%H.d38] {} {}
// CHECK:STDOUT:   %.decl.loc44: %.type = fn_decl @.1 [concrete = constants.%.d22] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.91d = fn_decl @H.1 [concrete = constants.%H.d38] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.2b6 = fn_decl @I.1 [concrete = constants.%I.c9a] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .G = file.%G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.2 {
// CHECK:STDOUT:   %G.decl: %G.type.bf6 = fn_decl @G.1 [concrete = constants.%G.e39] {} {}
// CHECK:STDOUT:   %H.decl: %H.type.e2f = fn_decl @H.2 [concrete = constants.%H.382] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.b27 = fn_decl @I.2 [concrete = constants.%I.a7f] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.a95
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
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
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

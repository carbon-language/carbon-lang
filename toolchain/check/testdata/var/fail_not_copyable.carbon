// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_not_copyable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_not_copyable.carbon

class X {
}

fn F(x: X) {
  // TODO: Strings should eventually be copyable, once we decide how to
  // represent them.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+4]]:19: error: cannot copy value of type `String` [CopyOfUncopyableType]
  // CHECK:STDERR:   var s: String = "hello";
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  var s: String = "hello";

  // TODO: Decide on rules for when classes are copyable.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+4]]:14: error: cannot copy value of type `X` [CopyOfUncopyableType]
  // CHECK:STDERR:   var y: X = x;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  var y: X = x;
}

// CHECK:STDOUT: --- fail_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %str: String = string_literal "hello" [concrete]
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
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %X = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref.loc14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: String = binding_pattern s
// CHECK:STDOUT:     %.loc21: String = var_pattern %s.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref String = var s
// CHECK:STDOUT:   %str: String = string_literal "hello" [concrete = constants.%str]
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %s: ref String = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %X = binding_pattern y
// CHECK:STDOUT:     %.loc28: %X = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %X = var y
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   %X.ref.loc28: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %y: ref %X = bind_name y, %y.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

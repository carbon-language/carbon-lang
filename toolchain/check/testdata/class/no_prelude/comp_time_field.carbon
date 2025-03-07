// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/comp_time_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/comp_time_field.carbon

// --- fail_let.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_let.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let A:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let A:! type = Class;

  // CHECK:STDERR: fail_let.carbon:[[@LINE+4]]:7: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:   let template B:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let template B:! type = Class;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_var.carbon:[[@LINE+8]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var C:! type = Class;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:16: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var C:! type = Class;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  var C:! type = Class;

  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:25: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var template D:! type = Class;
  // CHECK:STDERR:                         ^
  // CHECK:STDERR:
  var template D:! type = Class;
}

var x: Class = {};

// CHECK:STDOUT: --- fail_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt: type = binding_pattern A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %A: type = bind_name A, %Class.ref.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %B.patt: type = binding_pattern B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc15: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %B: type = bind_name B, %Class.ref.loc15
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .B = %B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/comp_time_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/comp_time_field.carbon

// --- fail_let.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_let.carbon:[[@LINE+2]]:15-21: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:
  let generic A: type = Class;

  // CHECK:STDERR: fail_let.carbon:[[@LINE+2]]:7-22: error: semantics TODO: ``let` compile time binding outside function or interface` [SemanticsTodo]
  // CHECK:STDERR:
  let template B: type = Class;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:15-21: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var.carbon:[[@LINE+2]]:23: error: found generic binding inside `var` pattern [NonRegularBindingInVarDecl]
  // CHECK:STDERR:
  var generic C: type = Class;

  // CHECK:STDERR: fail_var.carbon:[[@LINE+2]]:24: error: found generic binding inside `var` pattern [NonRegularBindingInVarDecl]
  // CHECK:STDERR:
  var template D: type = Class;
}

var x: Class = {};

// CHECK:STDOUT: --- fail_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.patt: %pattern_type = value_binding_pattern A [concrete]
// CHECK:STDOUT:   %B.patt: %pattern_type = value_binding_pattern B [concrete]
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
// CHECK:STDOUT:   %Class.ref.loc7: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: type = wrapper_binding A, %Class.ref.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt: %pattern_type = value_binding_pattern A [concrete = constants.%A.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc11: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc11_19.1: type = splice_block %.loc11_19.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc11: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc11_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: type = wrapper_binding B, %Class.ref.loc11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %B.patt: %pattern_type = value_binding_pattern B [concrete = constants.%B.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:

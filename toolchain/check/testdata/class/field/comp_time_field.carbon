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
  // CHECK:STDERR:        ╭─┤ fail_let.carbon:[[@LINE+9]]:15
  // CHECK:STDERR: ->     │   let generic A: type = Class;
  // CHECK:STDERR:        ·               ━━━┳━━━
  // CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━┛
  // CHECK:STDERR:        · ┃ error: semantics TODO:
  // CHECK:STDERR:        · ┃        ``let` compile time binding outside function or interface`
  // CHECK:STDERR:        · ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let generic A: type = Class;

  // CHECK:STDERR:        ╭─┤ fail_let.carbon:[[@LINE+8]]:7
  // CHECK:STDERR: ->     │   let template B: type = Class;
  // CHECK:STDERR:        ·       ┳━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: semantics TODO:
  // CHECK:STDERR:        ·          ┃        ``let` compile time binding outside function or interface`
  // CHECK:STDERR:        ·          ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let template B: type = Class;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR:        ╭─┤ fail_var.carbon:[[@LINE+15]]:15
  // CHECK:STDERR: ->     │   var generic C: type = Class;
  // CHECK:STDERR:        ·               ┳━━━━━━
  // CHECK:STDERR:        ·               ┗━━┫ error: semantics TODO:
  // CHECK:STDERR:        ·                  ┃        `handle invalid parse trees in `check``
  // CHECK:STDERR:        ·                  ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_var.carbon:[[@LINE+7]]:23
  // CHECK:STDERR: ->     │   var generic C: type = Class;
  // CHECK:STDERR:        ·                       ┳
  // CHECK:STDERR:        ·                       ┗━━┫ error: found generic binding inside `var` pattern
  // CHECK:STDERR:        ·                          ┃        [NonRegularBindingInVarDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var generic C: type = Class;

  // CHECK:STDERR:        ╭─┤ fail_var.carbon:[[@LINE+7]]:24
  // CHECK:STDERR: ->     │   var template D: type = Class;
  // CHECK:STDERR:        ·                        ┳
  // CHECK:STDERR:        ·                        ┗━━┫ error: found generic binding inside `var` pattern
  // CHECK:STDERR:        ·                           ┃        [NonRegularBindingInVarDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var template D: type = Class;
}

var x: Class = {};

// CHECK:STDOUT: --- fail_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
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
// CHECK:STDOUT:   %Class.ref.loc14: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc14_18.1: type = splice_block %.loc14_18.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc14: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc14_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: type = wrapper_binding A, %Class.ref.loc14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt: %pattern_type = value_binding_pattern A [concrete = constants.%A.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc24: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc24_19.1: type = splice_block %.loc24_19.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc24: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc24_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: type = wrapper_binding B, %Class.ref.loc24
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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

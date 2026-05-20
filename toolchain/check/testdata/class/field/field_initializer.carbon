// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/field_initializer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/field_initializer.carbon

// --- field_initializer.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class {
  var field: i32 = 123;
}

// CHECK:STDERR: field_initializer.carbon:[[@LINE+4]]:16: error: cannot initialize class with 1 field from struct with 0 fields [StructInitElementCountMismatch]
// CHECK:STDERR: var C: Class = {};
// CHECK:STDERR:                ^~
// CHECK:STDERR:
var C: Class = {};
//@dump-sem-ir-end

// --- fail_field_initializer_wrong_type.carbon
library "[[@TEST_NAME]]";

class Class {
  var field: i32 = ();
}

// --- fail_field_initializer_not_constant.carbon
library "[[@TEST_NAME]]";

var x: i32;
class Class {
  var field: i32 = x;
}

// CHECK:STDOUT: --- field_initializer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.d48: <witness> = complete_type_witness %struct_type.field [concrete]
// CHECK:STDOUT:   %pattern_type.904: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %C.patt: %pattern_type.904 = ref_binding_pattern C [concrete]
// CHECK:STDOUT:     %C.var_patt: %pattern_type.904 = var_pattern %C.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.var: ref %Class = var %C.var_patt [concrete]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %C: ref %Class = ref_binding C, %C.var [concrete = %C.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc5: %Class.elem = field_decl field, element0 [concrete]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.field [concrete = constants.%complete_type.d48]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .field = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%C.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

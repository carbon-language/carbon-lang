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

var C: Class = {};
//@dump-sem-ir-end

// --- field_initializer_import.carbon
library "[[@TEST_NAME]]";

import library "field_initializer";

var C2: Class = {};

// --- field_initializer_generic.carbon
library "[[@TEST_NAME]]";

interface I {
  let Default:! Self;
}

impl i32 as I where .Default = 123 {}

class Class(T:! I) {
  var x: T = T.Default;
}

var C: Class(i32) = {};

// --- fail_field_initializer_invalid_name.carbon
library "[[@TEST_NAME]]";

class Class {
  var field: i32 = 123;
}

// CHECK:STDERR: fail_field_initializer_invalid_name.carbon:[[@LINE+4]]:16: error: struct `Class` has no field named `not_a_field` [StructInitUnexpectedFieldInLiteral]
// CHECK:STDERR: var C: Class = {.not_a_field = 456};
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var C: Class = {.not_a_field = 456};

// --- fail_field_initializer_wrong_type.carbon
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_field_initializer_wrong_type.carbon:[[@LINE+7]]:20: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var field: i32 = ();
  // CHECK:STDERR:                    ^~
  // CHECK:STDERR: fail_field_initializer_wrong_type.carbon:[[@LINE+4]]:20: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var field: i32 = ();
  // CHECK:STDERR:                    ^~
  // CHECK:STDERR:
  var field: i32 = ();
}

// --- fail_field_initializer_not_constant.carbon
library "[[@TEST_NAME]]";

var x: i32;
class Class {
  // CHECK:STDERR: fail_field_initializer_not_constant.carbon:[[@LINE+4]]:20: error: semantics TODO: `field initializer is not constant` [SemanticsTodo]
  // CHECK:STDERR:   var field: i32 = x;
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  var field: i32 = x;
}

var C: Class = {};

// CHECK:STDOUT: --- field_initializer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.5b5: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c67: <witness> = complete_type_witness %struct_type.field [concrete]
// CHECK:STDOUT:   %pattern_type.415: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%int_123.5b5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %C.patt: %pattern_type.415 = ref_binding_pattern C [concrete]
// CHECK:STDOUT:     %C.var_patt: %pattern_type.415 = var_pattern %C.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.var: ref %Class = var %C.var_patt [concrete]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %C: ref %Class = ref_binding C, %C.var [concrete = %C.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc5_12: %Class.elem = field_decl field, element0, initializer = inst640000D4 [concrete]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc5_20.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_20.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_20.2(%int_123) [concrete = constants.%int_123.5b5]
// CHECK:STDOUT:   %.loc5_20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.5b5]
// CHECK:STDOUT:   %.loc5_20.2: %i32 = converted %int_123, %.loc5_20.1 [concrete = constants.%int_123.5b5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.field [concrete = constants.%complete_type.c67]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .field = %.loc5_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_17.2: init %Class to file.%C.var = class_init (constants.%int_123.5b5) [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc8_1: init %Class = converted %.loc8_17.1, %.loc8_17.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   assign file.%C.var, %.loc8_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

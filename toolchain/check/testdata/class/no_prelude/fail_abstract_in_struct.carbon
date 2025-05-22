// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/fail_abstract_in_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/fail_abstract_in_struct.carbon

// --- fail_abstract_field.carbon
library "[[@TEST_NAME]]";

abstract class Abstract1 {}

class Contains {
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE+7]]:10: error: field has abstract type `{.m1: Abstract1}` [AbstractTypeInFieldDecl]
  // CHECK:STDERR:   var a: {.m1: Abstract1};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract1 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: {.m1: Abstract1};
}

// --- fail_abstract_var.carbon
library "[[@TEST_NAME]]";

abstract class Abstract2 {}

// CHECK:STDERR: fail_abstract_var.carbon:[[@LINE+7]]:8: error: binding pattern has abstract type `{.m2: Abstract2}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v: {.m2: Abstract2};
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_var.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract2 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v: {.m2: Abstract2};

// --- abstract_let.carbon
library "[[@TEST_NAME]]";

abstract class Abstract3 {
}

fn F(a: Abstract3) {
  let l: {.m3: Abstract3} = {.m3 = a};
}

// --- fail_abstract_twice.carbon
library "[[@TEST_NAME]]";

abstract class Abstract4 {}
abstract class Abstract5 {}

// CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.m4: Abstract4, .m5: Abstract5}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v2: {.m4: Abstract4, .m5: Abstract5};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract4 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v2: {.m4: Abstract4, .m5: Abstract5};

// --- fail_abstract_first.carbon
library "[[@TEST_NAME]]";

abstract class Abstract6 {}

// CHECK:STDERR: fail_abstract_first.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.m6: Abstract6, .c1: ()}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v3: {.m6: Abstract6, .c1: ()};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_first.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract6 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v3: {.m6: Abstract6, .c1: ()};

// --- fail_abstract_second.carbon
library "[[@TEST_NAME]]";

abstract class Abstract7 {}

// CHECK:STDERR: fail_abstract_second.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.c2: (), .m7: Abstract7}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v4: {.c2: (), .m7: Abstract7};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_second.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract7 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v4: {.c2: (), .m7: Abstract7};

// --- lib.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}

// --- fail_import.carbon
library "[[@TEST_NAME]]";

import library "lib";

// CHECK:STDERR: fail_import.carbon:[[@LINE+8]]:9: error: binding pattern has abstract type `{.m: Abstract}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v5: {.m: Abstract};
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_import.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: lib.carbon:3:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v5: {.m: Abstract};

// CHECK:STDOUT: --- fail_abstract_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract1: type = class_type @Abstract1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Contains: type = class_type @Contains [concrete]
// CHECK:STDOUT:   %struct_type.m1.198: type = struct_type {.m1: %Abstract1} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract1 = %Abstract1.decl
// CHECK:STDOUT:     .Contains = %Contains.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract1.decl: type = class_decl @Abstract1 [concrete = constants.%Abstract1] {} {}
// CHECK:STDOUT:   %Contains.decl: type = class_decl @Contains [concrete = constants.%Contains] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract1 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Contains {
// CHECK:STDOUT:   %Abstract1.ref: type = name_ref Abstract1, file.%Abstract1.decl [concrete = constants.%Abstract1]
// CHECK:STDOUT:   %struct_type.m1: type = struct_type {.m1: %Abstract1} [concrete = constants.%struct_type.m1.198]
// CHECK:STDOUT:   %.loc13: <error> = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: <error>} [concrete = <error>]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Contains
// CHECK:STDOUT:   .Abstract1 = <poisoned>
// CHECK:STDOUT:   .a = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract2: type = class_type @Abstract2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.m2.155: type = struct_type {.m2: %Abstract2} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract2 = %Abstract2.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract2.decl: type = class_decl @Abstract2 [concrete = constants.%Abstract2] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: <error> = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var %v.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc12: type = splice_block %struct_type.m2 [concrete = constants.%struct_type.m2.155] {
// CHECK:STDOUT:     %Abstract2.ref: type = name_ref Abstract2, %Abstract2.decl [concrete = constants.%Abstract2]
// CHECK:STDOUT:     %struct_type.m2: type = struct_type {.m2: %Abstract2} [concrete = constants.%struct_type.m2.155]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract2 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract3: type = class_type @Abstract3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.32b: type = pattern_type %Abstract3 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.m3.fe4: type = struct_type {.m3: %Abstract3} [concrete]
// CHECK:STDOUT:   %pattern_type.a22: type = pattern_type %struct_type.m3.fe4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract3 = %Abstract3.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract3.decl: type = class_decl @Abstract3 [concrete = constants.%Abstract3] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.32b = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.32b = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %Abstract3 = value_param call_param0
// CHECK:STDOUT:     %Abstract3.ref.loc6: type = name_ref Abstract3, file.%Abstract3.decl [concrete = constants.%Abstract3]
// CHECK:STDOUT:     %a: %Abstract3 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract3 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %Abstract3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %l.patt: %pattern_type.a22 = binding_pattern l [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %Abstract3 = name_ref a, %a
// CHECK:STDOUT:   %.loc7_37: %struct_type.m3.fe4 = struct_literal (%a.ref)
// CHECK:STDOUT:   %.loc7_25: type = splice_block %struct_type.m3 [concrete = constants.%struct_type.m3.fe4] {
// CHECK:STDOUT:     %Abstract3.ref.loc7: type = name_ref Abstract3, file.%Abstract3.decl [concrete = constants.%Abstract3]
// CHECK:STDOUT:     %struct_type.m3: type = struct_type {.m3: %Abstract3} [concrete = constants.%struct_type.m3.fe4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %l: %struct_type.m3.fe4 = bind_name l, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_twice.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract4: type = class_type @Abstract4 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Abstract5: type = class_type @Abstract5 [concrete]
// CHECK:STDOUT:   %struct_type.m4.m5.c86: type = struct_type {.m4: %Abstract4, .m5: %Abstract5} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract4 = %Abstract4.decl
// CHECK:STDOUT:     .Abstract5 = %Abstract5.decl
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract4.decl: type = class_decl @Abstract4 [concrete = constants.%Abstract4] {} {}
// CHECK:STDOUT:   %Abstract5.decl: type = class_decl @Abstract5 [concrete = constants.%Abstract5] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: <error> = binding_pattern v2 [concrete]
// CHECK:STDOUT:     %v2.var_patt: <error> = var_pattern %v2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2.var: ref <error> = var %v2.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc13: type = splice_block %struct_type.m4.m5 [concrete = constants.%struct_type.m4.m5.c86] {
// CHECK:STDOUT:     %Abstract4.ref: type = name_ref Abstract4, %Abstract4.decl [concrete = constants.%Abstract4]
// CHECK:STDOUT:     %Abstract5.ref: type = name_ref Abstract5, %Abstract5.decl [concrete = constants.%Abstract5]
// CHECK:STDOUT:     %struct_type.m4.m5: type = struct_type {.m4: %Abstract4, .m5: %Abstract5} [concrete = constants.%struct_type.m4.m5.c86]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: <error> = bind_name v2, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract4 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract5 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_first.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract6: type = class_type @Abstract6 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.m6.c1.489: type = struct_type {.m6: %Abstract6, .c1: %empty_tuple.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract6 = %Abstract6.decl
// CHECK:STDOUT:     .v3 = %v3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract6.decl: type = class_decl @Abstract6 [concrete = constants.%Abstract6] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v3.patt: <error> = binding_pattern v3 [concrete]
// CHECK:STDOUT:     %v3.var_patt: <error> = var_pattern %v3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3.var: ref <error> = var %v3.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc12_33: type = splice_block %struct_type.m6.c1 [concrete = constants.%struct_type.m6.c1.489] {
// CHECK:STDOUT:     %Abstract6.ref: type = name_ref Abstract6, %Abstract6.decl [concrete = constants.%Abstract6]
// CHECK:STDOUT:     %.loc12_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_32.2: type = converted %.loc12_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.m6.c1: type = struct_type {.m6: %Abstract6, .c1: %empty_tuple.type} [concrete = constants.%struct_type.m6.c1.489]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3: <error> = bind_name v3, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract6 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_second.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract7: type = class_type @Abstract7 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.c2.m7.4a7: type = struct_type {.c2: %empty_tuple.type, .m7: %Abstract7} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract7 = %Abstract7.decl
// CHECK:STDOUT:     .v4 = %v4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract7.decl: type = class_decl @Abstract7 [concrete = constants.%Abstract7] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v4.patt: <error> = binding_pattern v4 [concrete]
// CHECK:STDOUT:     %v4.var_patt: <error> = var_pattern %v4.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4.var: ref <error> = var %v4.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc12_33: type = splice_block %struct_type.c2.m7 [concrete = constants.%struct_type.c2.m7.4a7] {
// CHECK:STDOUT:     %.loc12_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_16.2: type = converted %.loc12_16.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Abstract7.ref: type = name_ref Abstract7, %Abstract7.decl [concrete = constants.%Abstract7]
// CHECK:STDOUT:     %struct_type.c2.m7: type = struct_type {.c2: %empty_tuple.type, .m7: %Abstract7} [concrete = constants.%struct_type.c2.m7.4a7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4: <error> = bind_name v4, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract7 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [concrete = constants.%Abstract] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.m.50b: type = struct_type {.m: %Abstract} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Abstract: type = import_ref Main//lib, Abstract, loaded [concrete = constants.%Abstract]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//lib, loc3_26, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee1 = import_ref Main//lib, inst16 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract = imports.%Main.Abstract
// CHECK:STDOUT:     .v5 = %v5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v5.patt: <error> = binding_pattern v5 [concrete]
// CHECK:STDOUT:     %v5.var_patt: <error> = var_pattern %v5.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5.var: ref <error> = var %v5.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc13: type = splice_block %struct_type.m [concrete = constants.%struct_type.m.50b] {
// CHECK:STDOUT:     %Abstract.ref: type = name_ref Abstract, imports.%Main.Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %struct_type.m: type = struct_type {.m: %Abstract} [concrete = constants.%struct_type.m.50b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5: <error> = bind_name v5, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract [from "lib.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee1
// CHECK:STDOUT: }
// CHECK:STDOUT:

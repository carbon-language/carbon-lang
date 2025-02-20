// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/todo_fail_abstract_in_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/todo_fail_abstract_in_struct.carbon

// --- todo_fail_abstract_field.carbon
library "[[@TEST_NAME]]";

abstract class Abstract1 {}

class Contains {
  var a: {.m1: Abstract1};
}

// --- todo_fail_abstract_var.carbon
library "[[@TEST_NAME]]";

abstract class Abstract2 {}

fn Var() {
  var v: {.m2: Abstract2};
}

// --- abstract_let.carbon
library "[[@TEST_NAME]]";

abstract class Abstract3 {
}

fn F(a: Abstract3) {
  let l: {.m3: Abstract3} = {.m3 = a};
}

// --- todo_fail_abstract_twice.carbon
library "[[@TEST_NAME]]";

abstract class Abstract4 {}
abstract class Abstract5 {}

fn Var2() {
  var v2: {.m4: Abstract4, .m5: Abstract5};
}

// --- todo_fail_abstract_first.carbon
library "[[@TEST_NAME]]";

abstract class Abstract6 {}

fn Var3() {
  var v3: {.m6: Abstract6, .c1: ()};
}

// --- todo_fail_abstract_second.carbon
library "[[@TEST_NAME]]";

abstract class Abstract7 {}

fn Var4() {
  var v4: {.c2: (), .m7: Abstract7};
}

// --- lib.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}

// --- todo_fail_import.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Var5() {
  var v5: {.m: Abstract};
}

// CHECK:STDOUT: --- todo_fail_abstract_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract1: type = class_type @Abstract1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Contains: type = class_type @Contains [concrete]
// CHECK:STDOUT:   %struct_type.m1.198: type = struct_type {.m1: %Abstract1} [concrete]
// CHECK:STDOUT:   %Contains.elem: type = unbound_element_type %Contains, %struct_type.m1.198 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %struct_type.m1.198} [concrete]
// CHECK:STDOUT:   %complete_type.be2: <witness> = complete_type_witness %struct_type.a [concrete]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Contains {
// CHECK:STDOUT:   %.loc6_8: %Contains.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %Contains.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Contains.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.be2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Contains
// CHECK:STDOUT:   .Abstract1 = <poisoned>
// CHECK:STDOUT:   .a = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_abstract_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract2: type = class_type @Abstract2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [concrete]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.m2.155: type = struct_type {.m2: %Abstract2} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract2 = %Abstract2.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract2.decl: type = class_decl @Abstract2 [concrete = constants.%Abstract2] {} {}
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [concrete = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %struct_type.m2.155 = binding_pattern v
// CHECK:STDOUT:     %.loc6_3: %struct_type.m2.155 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %struct_type.m2.155 = var v
// CHECK:STDOUT:   %.loc6_25: type = splice_block %struct_type.m2 [concrete = constants.%struct_type.m2.155] {
// CHECK:STDOUT:     %Abstract2.ref: type = name_ref Abstract2, file.%Abstract2.decl [concrete = constants.%Abstract2]
// CHECK:STDOUT:     %struct_type.m2: type = struct_type {.m2: %Abstract2} [concrete = constants.%struct_type.m2.155]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %struct_type.m2.155 = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract3: type = class_type @Abstract3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.m3.fe4: type = struct_type {.m3: %Abstract3} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract3 = %Abstract3.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract3.decl: type = class_decl @Abstract3 [concrete = constants.%Abstract3] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %Abstract3 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Abstract3 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %Abstract3 = value_param runtime_param0
// CHECK:STDOUT:     %Abstract3.ref.loc6: type = name_ref Abstract3, file.%Abstract3.decl [concrete = constants.%Abstract3]
// CHECK:STDOUT:     %a: %Abstract3 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract3 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %Abstract3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %l.patt: %struct_type.m3.fe4 = binding_pattern l
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
// CHECK:STDOUT: --- todo_fail_abstract_twice.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract4: type = class_type @Abstract4 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Abstract5: type = class_type @Abstract5 [concrete]
// CHECK:STDOUT:   %Var2.type: type = fn_type @Var2 [concrete]
// CHECK:STDOUT:   %Var2: %Var2.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.m4.m5.c86: type = struct_type {.m4: %Abstract4, .m5: %Abstract5} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract4 = %Abstract4.decl
// CHECK:STDOUT:     .Abstract5 = %Abstract5.decl
// CHECK:STDOUT:     .Var2 = %Var2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract4.decl: type = class_decl @Abstract4 [concrete = constants.%Abstract4] {} {}
// CHECK:STDOUT:   %Abstract5.decl: type = class_decl @Abstract5 [concrete = constants.%Abstract5] {} {}
// CHECK:STDOUT:   %Var2.decl: %Var2.type = fn_decl @Var2 [concrete = constants.%Var2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %struct_type.m4.m5.c86 = binding_pattern v2
// CHECK:STDOUT:     %.loc7_3: %struct_type.m4.m5.c86 = var_pattern %v2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2.var: ref %struct_type.m4.m5.c86 = var v2
// CHECK:STDOUT:   %.loc7_42: type = splice_block %struct_type.m4.m5 [concrete = constants.%struct_type.m4.m5.c86] {
// CHECK:STDOUT:     %Abstract4.ref: type = name_ref Abstract4, file.%Abstract4.decl [concrete = constants.%Abstract4]
// CHECK:STDOUT:     %Abstract5.ref: type = name_ref Abstract5, file.%Abstract5.decl [concrete = constants.%Abstract5]
// CHECK:STDOUT:     %struct_type.m4.m5: type = struct_type {.m4: %Abstract4, .m5: %Abstract5} [concrete = constants.%struct_type.m4.m5.c86]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: ref %struct_type.m4.m5.c86 = bind_name v2, %v2.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_abstract_first.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract6: type = class_type @Abstract6 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Var3.type: type = fn_type @Var3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Var3: %Var3.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.m6.c1.489: type = struct_type {.m6: %Abstract6, .c1: %empty_tuple.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract6 = %Abstract6.decl
// CHECK:STDOUT:     .Var3 = %Var3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract6.decl: type = class_decl @Abstract6 [concrete = constants.%Abstract6] {} {}
// CHECK:STDOUT:   %Var3.decl: %Var3.type = fn_decl @Var3 [concrete = constants.%Var3] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract6 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v3.patt: %struct_type.m6.c1.489 = binding_pattern v3
// CHECK:STDOUT:     %.loc6_3: %struct_type.m6.c1.489 = var_pattern %v3.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3.var: ref %struct_type.m6.c1.489 = var v3
// CHECK:STDOUT:   %.loc6_35: type = splice_block %struct_type.m6.c1 [concrete = constants.%struct_type.m6.c1.489] {
// CHECK:STDOUT:     %Abstract6.ref: type = name_ref Abstract6, file.%Abstract6.decl [concrete = constants.%Abstract6]
// CHECK:STDOUT:     %.loc6_34.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_34.2: type = converted %.loc6_34.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.m6.c1: type = struct_type {.m6: %Abstract6, .c1: %empty_tuple.type} [concrete = constants.%struct_type.m6.c1.489]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v3: ref %struct_type.m6.c1.489 = bind_name v3, %v3.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_abstract_second.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract7: type = class_type @Abstract7 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Var4.type: type = fn_type @Var4 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Var4: %Var4.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.c2.m7.4a7: type = struct_type {.c2: %empty_tuple.type, .m7: %Abstract7} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract7 = %Abstract7.decl
// CHECK:STDOUT:     .Var4 = %Var4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract7.decl: type = class_decl @Abstract7 [concrete = constants.%Abstract7] {} {}
// CHECK:STDOUT:   %Var4.decl: %Var4.type = fn_decl @Var4 [concrete = constants.%Var4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract7 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var4() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v4.patt: %struct_type.c2.m7.4a7 = binding_pattern v4
// CHECK:STDOUT:     %.loc6_3: %struct_type.c2.m7.4a7 = var_pattern %v4.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4.var: ref %struct_type.c2.m7.4a7 = var v4
// CHECK:STDOUT:   %.loc6_35: type = splice_block %struct_type.c2.m7 [concrete = constants.%struct_type.c2.m7.4a7] {
// CHECK:STDOUT:     %.loc6_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_18.2: type = converted %.loc6_18.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Abstract7.ref: type = name_ref Abstract7, file.%Abstract7.decl [concrete = constants.%Abstract7]
// CHECK:STDOUT:     %struct_type.c2.m7: type = struct_type {.c2: %empty_tuple.type, .m7: %Abstract7} [concrete = constants.%struct_type.c2.m7.4a7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v4: ref %struct_type.c2.m7.4a7 = bind_name v4, %v4.var
// CHECK:STDOUT:   return
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Var5.type: type = fn_type @Var5 [concrete]
// CHECK:STDOUT:   %Var5: %Var5.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.m.50b: type = struct_type {.m: %Abstract} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Abstract: type = import_ref Main//lib, Abstract, loaded [concrete = constants.%Abstract]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//lib, loc3_26, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee1 = import_ref Main//lib, inst14 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Abstract = imports.%Main.Abstract
// CHECK:STDOUT:     .Var5 = %Var5.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Var5.decl: %Var5.type = fn_decl @Var5 [concrete = constants.%Var5] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract [from "lib.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v5.patt: %struct_type.m.50b = binding_pattern v5
// CHECK:STDOUT:     %.loc6_3: %struct_type.m.50b = var_pattern %v5.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5.var: ref %struct_type.m.50b = var v5
// CHECK:STDOUT:   %.loc6_24: type = splice_block %struct_type.m [concrete = constants.%struct_type.m.50b] {
// CHECK:STDOUT:     %Abstract.ref: type = name_ref Abstract, imports.%Main.Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %struct_type.m: type = struct_type {.m: %Abstract} [concrete = constants.%struct_type.m.50b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v5: ref %struct_type.m.50b = bind_name v5, %v5.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

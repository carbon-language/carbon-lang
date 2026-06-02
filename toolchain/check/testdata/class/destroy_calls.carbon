// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/destroy_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/destroy_calls.carbon

// --- implicit_return.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

//@dump-sem-ir-begin
fn F() {
  var unused a: A = {};
  var unused b: B = {};
  var unused c: C = {};
}
//@dump-sem-ir-end

// --- nested_scope.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

//@dump-sem-ir-begin
fn F() {
  var unused a: A = {};
  var unused b: B = {};
  if (true) {
    var unused c: C = {};
  }
}
//@dump-sem-ir-end

// --- temp.carbon

library "[[@TEST_NAME]]";

class A { fn Make() -> A; }
class B { fn Make() -> B; }
class C { fn Make() -> C; }

//@dump-sem-ir-begin
fn F() {
  // TODO: The scoping of these destroy calls is incorrect. Maybe we need to
  // establish statement scopes?
  A.Make();
  B.Make();
  C.Make();
}
//@dump-sem-ir-end

// --- generic_class.carbon
library "[[@TEST_NAME]]";

class C {}

class D(template T:! type) {}

//@dump-sem-ir-begin
fn F() {
  var unused a: D(C) = {};
}
//@dump-sem-ir-end

// --- generic_use_inside_generic.carbon
library "[[@TEST_NAME]]";

class C(template T:! type) {}

//@dump-sem-ir-begin
fn F(template T:! type) {
  var unused v: C(T) = {};
}
//@dump-sem-ir-end

fn G() { F({}); }

// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ccb: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.787: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ccb = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ccb = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.787 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.787 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = ref_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0d5 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.0d5 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc12_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_22.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_3: init %C = converted %.loc12_22.1, %.loc12_22.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc12_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %c.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ccb: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.787: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc13_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ccb = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ccb = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.787 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.787 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = ref_binding b, %b.var
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.0d5 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.0d5 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_24.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_5: init %C = converted %.loc13_24.1, %.loc13_24.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc13_5
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %c.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %b.var, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_5.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.Make.type: type = fn_type @A.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.Make: %A.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.Make.type: type = fn_type @B.Make [concrete]
// CHECK:STDOUT:   %B.Make: %B.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.Make.type: type = fn_type @C.Make [concrete]
// CHECK:STDOUT:   %C.Make: %C.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc14_10.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.3: type = fn_type @Destroy.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.3: %Destroy.Op.type.790b3b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref.loc12: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc12_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %A.Make.call: init %A to %.loc12_10.1 = call %Make.ref.loc12()
// CHECK:STDOUT:   %.loc12_10.2: ref %A = temporary %.loc12_10.1, %A.Make.call
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Make.ref.loc13: %B.Make.type = name_ref Make, @B.%B.Make.decl [concrete = constants.%B.Make]
// CHECK:STDOUT:   %.loc13_10.1: ref %B = temporary_storage
// CHECK:STDOUT:   %B.Make.call: init %B to %.loc13_10.1 = call %Make.ref.loc13()
// CHECK:STDOUT:   %.loc13_10.2: ref %B = temporary %.loc13_10.1, %B.Make.call
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Make.ref.loc14: %C.Make.type = name_ref Make, @C.%C.Make.decl [concrete = constants.%C.Make]
// CHECK:STDOUT:   %.loc14_10.1: ref %C = temporary_storage
// CHECK:STDOUT:   %C.Make.call: init %C to %.loc14_10.1 = call %Make.ref.loc14()
// CHECK:STDOUT:   %.loc14_10.2: ref %C = temporary %.loc14_10.1, %C.Make.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc14: <bound method> = bound_method %.loc14_10.2, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call.loc14: init %empty_tuple.type = call %Destroy.Op.bound.loc14(%.loc14_10.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_10.2, constants.%Destroy.Op.2dd605.3
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_10.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %.loc12_10.2, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%.loc12_10.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_10.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.66a: type = class_type @D, @D(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.855: type = pattern_type %D.66a [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D.66a = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.855 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.855 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %D.66a = var %a.var_patt
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_25.2: init %D.66a to %a.var = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc9_3: init %D.66a = converted %.loc9_25.1, %.loc9_25.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %D [concrete = constants.%D.66a] {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%C) [concrete = constants.%D.66a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %D.66a = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %D.66a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1f9: type = class_type @C, @C(%T) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.1f9 [template]
// CHECK:STDOUT:   %pattern_type.69e: type = pattern_type %C.1f9 [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val.ad6: %C.1f9 = struct_value () [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.1f9, @Destroy [template]
// CHECK:STDOUT:   %Destroy.facet.61a: %Destroy.type = facet_value %C.1f9, (%Destroy.lookup_impl_witness) [template]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.269: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.61a) [template]
// CHECK:STDOUT:   %.320: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.269, %Destroy.facet.61a [template]
// CHECK:STDOUT:   %impl.elem0: %.320 = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.61a) [template]
// CHECK:STDOUT:   %C.ea5: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.4ac: type = pattern_type %C.ea5 [concrete]
// CHECK:STDOUT:   %C.val.986: %C.ea5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.2: <witness> = custom_witness (%Destroy.Op.2dd605.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.3e3: %Destroy.type = facet_value %C.ea5, (%custom_witness.e20f1f.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e1c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.3e3) [concrete]
// CHECK:STDOUT:   %.f04: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e1c, %Destroy.facet.3e3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_19.1: type = splice_block %.loc6_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2: type = class_type @C, @C(%T.loc6_16.1) [template = %C.loc7_20.2 (constants.%C.1f9)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc7_20.2 [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc7_20.2 [template = %pattern_type (constants.%pattern_type.69e)]
// CHECK:STDOUT:   %C.val: @F.%C.loc7_20.2 (%C.1f9) = struct_value () [template = %C.val (constants.%C.val.ad6)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc7_20.2, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc7_20.2, (%Destroy.lookup_impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.61a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [template = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.269)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [template = %.loc7_3.2 (constants.%.320)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.2 (%.320) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [template = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.69e) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.69e) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_20.2 (%C.1f9) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_25.2: init @F.%C.loc7_20.2 (%C.1f9) to %v.var = class_init () [template = %C.val (constants.%C.val.ad6)]
// CHECK:STDOUT:     %.loc7_3.1: init @F.%C.loc7_20.2 (%C.1f9) = converted %.loc7_25.1, %.loc7_25.2 [template = %C.val (constants.%C.val.ad6)]
// CHECK:STDOUT:     assign %v.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_20: type = splice_block %C.loc7_20.1 [template = %C.loc7_20.2 (constants.%C.1f9)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_20.1: type = class_type @C, @C(constants.%T) [template = %C.loc7_20.2 (constants.%C.1f9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_20.2 (%C.1f9) = ref_binding v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.2 (%.320) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.61a) [template = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %C.ea5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.ea5
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ac
// CHECK:STDOUT:   %C.val => constants.%C.val.986
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.2
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.3e3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e1c
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.f04
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
  A.Make();
  B.Make();
  C.Make();
}
//@dump-sem-ir-end

// --- generic_class.carbon
library "[[@TEST_NAME]]";

class C {}

class D(template T: type) {}

//@dump-sem-ir-begin
fn F() {
  var unused a: D(C) = {};
}
//@dump-sem-ir-end

// --- generic_use_inside_generic.carbon
library "[[@TEST_NAME]]";

class C(T: type) {}

//@dump-sem-ir-begin
fn F(generic T: type) {
  var unused v: C(T) = {};
}
//@dump-sem-ir-end

fn G() { F({}); }

// --- generic_use_inside_template.carbon
library "[[@TEST_NAME]]";

class C(template T: type) {}

//@dump-sem-ir-begin
fn F(template T: type) {
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
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc12_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_22.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_3: init %C = converted %.loc12_22.1, %.loc12_22.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc12_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.4
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
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc13_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_24.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_5: init %C = converted %.loc13_24.1, %.loc13_24.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc13_5
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%c.var)
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.4
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref.loc10: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc10_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %A.Make.call: init %A to %.loc10_10.1 = call %Make.ref.loc10()
// CHECK:STDOUT:   %.loc10_10.2: ref %A = temporary %.loc10_10.1, %A.Make.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %.loc10_10.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%.loc10_10.2)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Make.ref.loc11: %B.Make.type = name_ref Make, @B.%B.Make.decl [concrete = constants.%B.Make]
// CHECK:STDOUT:   %.loc11_10.1: ref %B = temporary_storage
// CHECK:STDOUT:   %B.Make.call: init %B to %.loc11_10.1 = call %Make.ref.loc11()
// CHECK:STDOUT:   %.loc11_10.2: ref %B = temporary %.loc11_10.1, %B.Make.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_10.2, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_10.2)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Make.ref.loc12: %C.Make.type = name_ref Make, @C.%C.Make.decl [concrete = constants.%C.Make]
// CHECK:STDOUT:   %.loc12_10.1: ref %C = temporary_storage
// CHECK:STDOUT:   %C.Make.call: init %C to %.loc12_10.1 = call %Make.ref.loc12()
// CHECK:STDOUT:   %.loc12_10.2: ref %C = temporary %.loc12_10.1, %C.Make.call
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %.loc12_10.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%.loc12_10.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_10.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12(%self.param: ref %C) {
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
// CHECK:STDOUT:   %D.dee: type = class_type @D, @D(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.6fa: type = pattern_type %D.dee [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6fa = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.6fa = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D.dee = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %D.dee = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_25.2: init %D.dee to %a.var = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc9_3: init %D.dee = converted %.loc9_25.1, %.loc9_25.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %D [concrete = constants.%D.dee] {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%C) [concrete = constants.%D.dee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %D.dee = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6fa = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6fa = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %D.dee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1d0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.1d0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ebe: type = pattern_type %C.1d0 [symbolic]
// CHECK:STDOUT:   %v.patt.f9d: %pattern_type.ebe = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt.ca3: %pattern_type.ebe = var_pattern %v.patt.f9d [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val.d4f: %C.1d0 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.1d0, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.19f: %Destroy.type = facet_value %C.1d0, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fc4: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.19f) [symbolic]
// CHECK:STDOUT:   %.cb3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fc4, %Destroy.facet.19f [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.cb3 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.WithSelf.Op(%Destroy.facet.19f) [symbolic]
// CHECK:STDOUT:   %C.d8e: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.2fd: type = pattern_type %C.d8e [concrete]
// CHECK:STDOUT:   %v.patt.0e7: %pattern_type.2fd = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.259: %pattern_type.2fd = var_pattern %v.patt.0e7 [concrete]
// CHECK:STDOUT:   %C.val.58e: %C.d8e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.da3: %Destroy.type = facet_value %C.d8e, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.294: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.da3) [concrete]
// CHECK:STDOUT:   %.98c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.294, %Destroy.facet.da3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2: type = class_type @C, @C(%T.loc6_15.1) [symbolic = %C.loc7_20.2 (constants.%C.1d0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc7_20.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc7_20.2 [symbolic = %pattern_type (constants.%pattern_type.ebe)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type (%pattern_type.ebe) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt.f9d)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type (%pattern_type.ebe) = var_pattern %v.patt.loc7_15.2 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt.ca3)]
// CHECK:STDOUT:   %C.val: @F.%C.loc7_20.2 (%C.1d0) = struct_value () [symbolic = %C.val (constants.%C.val.d4f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc7_20.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %C.loc7_20.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.19f)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.fc4)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.4 (constants.%.cb3)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.4 (%.cb3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_20.2 (%C.1d0) = var_storage %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_25.2: init @F.%C.loc7_20.2 (%C.1d0) to %v.var = class_init () [symbolic = %C.val (constants.%C.val.d4f)]
// CHECK:STDOUT:     %.loc7_3.1: init @F.%C.loc7_20.2 (%C.1d0) = converted %.loc7_25.1, %.loc7_25.2 [symbolic = %C.val (constants.%C.val.d4f)]
// CHECK:STDOUT:     assign %v.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_20: type = splice_block %C.loc7_20.1 [symbolic = %C.loc7_20.2 (constants.%C.1d0)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_15.2 [symbolic = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_20.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_20.2 (constants.%C.1d0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_20.2 (%C.1d0) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type (%pattern_type.ebe) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt.f9d)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type (%pattern_type.ebe) = var_pattern %v.patt.loc7_15.1 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt.ca3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.4 (%.cb3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%C.1d0, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.19f)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%C.1d0, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.19f)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%C.1d0, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.19f)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%C.1d0, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet.19f)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.19f) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%v.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %C.d8e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.d8e
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2fd
// CHECK:STDOUT:   %v.patt.loc7_15.2 => constants.%v.patt.0e7
// CHECK:STDOUT:   %v.var_patt.loc7_3.2 => constants.%v.var_patt.259
// CHECK:STDOUT:   %C.val => constants.%C.val.58e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.2
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3 => constants.%Destroy.facet.da3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.294
// CHECK:STDOUT:   %.loc7_3.4 => constants.%.98c
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.eec: type = splice_inst @F.%.loc7_20.6 [template]
// CHECK:STDOUT:   %.b01: type = type_of_inst @F.%.loc7_20.8 [template]
// CHECK:STDOUT:   %.195: %.b01 = splice_inst @F.%.loc7_20.8 [template]
// CHECK:STDOUT:   %.53f: type = splice_inst @F.%.loc7_20.11 [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.53f [template]
// CHECK:STDOUT:   %pattern_type.7d1: type = pattern_type %.53f [template]
// CHECK:STDOUT:   %v.patt.7b6: %pattern_type.7d1 = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.067: %pattern_type.7d1 = var_pattern %v.patt.7b6 [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.48a6b8.1: type = type_of_inst @F.%.loc7_3.17 [template]
// CHECK:STDOUT:   %.88d2c8.1: %.48a6b8.1 = splice_inst @F.%.loc7_3.17 [template]
// CHECK:STDOUT:   %.ceb5a2.1: type = type_of_inst @F.%.loc7_3.20 [template]
// CHECK:STDOUT:   %.45c659.1: %.ceb5a2.1 = splice_inst @F.%.loc7_3.20 [template]
// CHECK:STDOUT:   %.40e: type = type_of_inst @F.%.loc7_3.23 [template]
// CHECK:STDOUT:   %.fa1: %.40e = splice_inst @F.%.loc7_3.23 [template]
// CHECK:STDOUT:   %.1fa: type = type_of_inst @F.%.loc7_3.26 [template]
// CHECK:STDOUT:   %.e13: %.1fa = splice_inst @F.%.loc7_3.26 [template]
// CHECK:STDOUT:   %.6ec: %.53f = splice_inst @F.%.loc7_3.29 [template]
// CHECK:STDOUT:   %.817: %.53f = splice_inst @F.%.loc7_3.31 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.14f: %.53f = splice_inst @F.%.loc7_3.33 [template]
// CHECK:STDOUT:   %.c4e: type = type_of_inst @F.%.loc7_3.35 [template]
// CHECK:STDOUT:   %.869: %.c4e = splice_inst @F.%.loc7_3.35 [template]
// CHECK:STDOUT:   %.a78: type = type_of_inst @F.%.loc7_3.38 [template]
// CHECK:STDOUT:   %.3d1: %.a78 = splice_inst @F.%.loc7_3.38 [template]
// CHECK:STDOUT:   %inst.specific_inst.c72: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.62d: type = specific_inst %empty_struct_type, @F(%empty_struct_type) [concrete = %empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.d8efcc.1: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %inst.class_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %C.d8efcc.2: type = class_type @C, @C(%empty_struct_type) [concrete = %C.d8efcc.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.1b2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6eb: type = splice_block %C.d8efcc.1 [concrete = %C.d8efcc.1] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pattern_type.2fd: type = pattern_type %C.d8efcc.1 [concrete]
// CHECK:STDOUT:   %v.patt.0e7: %pattern_type.2fd = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.259: %pattern_type.2fd = var_pattern %v.patt.0e7 [concrete]
// CHECK:STDOUT:   %.0a6: <instruction> = call_action (%ImplicitAs.generic, %.53f), false [template]
// CHECK:STDOUT:   %.48a6b8.2: type = type_of_inst %.0a6 [template]
// CHECK:STDOUT:   %.88d2c8.2: %.48a6b8.2 = splice_inst %.0a6 [template]
// CHECK:STDOUT:   %.53c: <instruction> = access_member_action %.88d2c8.2, Convert [template]
// CHECK:STDOUT:   %.ceb5a2.2: type = type_of_inst %.53c [template]
// CHECK:STDOUT:   %.45c659.2: %.ceb5a2.2 = splice_inst %.53c [template]
// CHECK:STDOUT:   %.1e2: <instruction> = compound_member_access_action %empty_struct, %.45c659.2 [template]
// CHECK:STDOUT:   %.4f9: type = type_of_inst %.1e2 [template]
// CHECK:STDOUT:   %.bc1: %.4f9 = splice_inst %.1e2 [template]
// CHECK:STDOUT:   %.1ec: <instruction> = call_action (%.bc1), true [template]
// CHECK:STDOUT:   %.65e: type = type_of_inst %.1ec [template]
// CHECK:STDOUT:   %.fc2: %.65e = splice_inst %.1ec [template]
// CHECK:STDOUT:   %.e48: <instruction> = refine_inst_action %.fc2 [template]
// CHECK:STDOUT:   %.118: %C.d8efcc.1 = splice_inst %.e48 [template]
// CHECK:STDOUT:   %.663: <instruction> = convert_to_category_action %.118, element10 [template]
// CHECK:STDOUT:   %.e51: %C.d8efcc.1 = splice_inst %.663 [template]
// CHECK:STDOUT:   %inst.specific_inst.2e8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d94: ref %C.d8efcc.1 = specific_inst @F.%v.var, @F(%empty_struct_type)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.da3: %Destroy.type = facet_value %C.d8efcc.1, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.294: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.da3) [concrete]
// CHECK:STDOUT:   %.98c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.294, %Destroy.facet.da3 [concrete]
// CHECK:STDOUT:   %inst.splice_block.5e8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b36: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       %impl.elem0: %.98c = impl_witness_access %custom_witness.df9cc1.2, element0 [concrete = %Destroy.Op.1a2547.2]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.d94, %impl.elem0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %.b36(%.d94)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_20.6: <instruction> = refine_inst_action %T.ref [template]
// CHECK:STDOUT:   %.loc7_20.7: type = splice_inst %.loc7_20.6 [template = %.loc7_20.7 (constants.%.eec)]
// CHECK:STDOUT:   %.loc7_20.8: <instruction> = call_action (%C.ref, %.loc7_20.2), false [template]
// CHECK:STDOUT:   %.loc7_20.9: type = type_of_inst %.loc7_20.8 [template = %.loc7_20.9 (constants.%.b01)]
// CHECK:STDOUT:   %.loc7_20.10: @F.%.loc7_20.9 (%.b01) = splice_inst %.loc7_20.8 [template = %.loc7_20.10 (constants.%.195)]
// CHECK:STDOUT:   %.loc7_20.11: <instruction> = convert_to_value_action %.loc7_20.4, type [template]
// CHECK:STDOUT:   %.loc7_20.12: type = splice_inst %.loc7_20.11 [template = %.loc7_20.12 (constants.%.53f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc7_20.12 [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc7_20.12 [template = %pattern_type (constants.%pattern_type.7d1)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type (%pattern_type.7d1) = ref_binding_pattern v [template = %v.patt.loc7_15.2 (constants.%v.patt.7b6)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type (%pattern_type.7d1) = var_pattern %v.patt.loc7_15.2 [template = %v.var_patt.loc7_3.2 (constants.%v.var_patt.067)]
// CHECK:STDOUT:   %.loc7_3.17: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.53f), false [template]
// CHECK:STDOUT:   %.loc7_3.18: type = type_of_inst %.loc7_3.17 [template = %.loc7_3.18 (constants.%.48a6b8.1)]
// CHECK:STDOUT:   %.loc7_3.19: @F.%.loc7_3.18 (%.48a6b8.1) = splice_inst %.loc7_3.17 [template = %.loc7_3.19 (constants.%.88d2c8.1)]
// CHECK:STDOUT:   %.loc7_3.20: <instruction> = access_member_action %.loc7_3.2, Convert [template]
// CHECK:STDOUT:   %.loc7_3.21: type = type_of_inst %.loc7_3.20 [template = %.loc7_3.21 (constants.%.ceb5a2.1)]
// CHECK:STDOUT:   %.loc7_3.22: @F.%.loc7_3.21 (%.ceb5a2.1) = splice_inst %.loc7_3.20 [template = %.loc7_3.22 (constants.%.45c659.1)]
// CHECK:STDOUT:   %.loc7_3.23: <instruction> = compound_member_access_action %.loc7_25, %.loc7_3.4 [template]
// CHECK:STDOUT:   %.loc7_3.24: type = type_of_inst %.loc7_3.23 [template = %.loc7_3.24 (constants.%.40e)]
// CHECK:STDOUT:   %.loc7_3.25: @F.%.loc7_3.24 (%.40e) = splice_inst %.loc7_3.23 [template = %.loc7_3.25 (constants.%.fa1)]
// CHECK:STDOUT:   %.loc7_3.26: <instruction> = call_action (%.loc7_3.6), true [template]
// CHECK:STDOUT:   %.loc7_3.27: type = type_of_inst %.loc7_3.26 [template = %.loc7_3.27 (constants.%.1fa)]
// CHECK:STDOUT:   %.loc7_3.28: @F.%.loc7_3.27 (%.1fa) = splice_inst %.loc7_3.26 [template = %.loc7_3.28 (constants.%.e13)]
// CHECK:STDOUT:   %.loc7_3.29: <instruction> = refine_inst_action %.loc7_3.9 [template]
// CHECK:STDOUT:   %.loc7_3.30: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.29 [template = %.loc7_3.30 (constants.%.6ec)]
// CHECK:STDOUT:   %.loc7_3.31: <instruction> = convert_to_category_action %.loc7_3.10, element10 [template]
// CHECK:STDOUT:   %.loc7_3.32: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.31 [template = %.loc7_3.32 (constants.%.817)]
// CHECK:STDOUT:   %.loc7_3.33: <instruction> = refine_inst_action %v.var [template]
// CHECK:STDOUT:   %.loc7_3.34: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.33 [template = %.loc7_3.34 (constants.%.14f)]
// CHECK:STDOUT:   %.loc7_3.35: <instruction> = compound_member_access_action %.loc7_3.12, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc7_3.36: type = type_of_inst %.loc7_3.35 [template = %.loc7_3.36 (constants.%.c4e)]
// CHECK:STDOUT:   %.loc7_3.37: @F.%.loc7_3.36 (%.c4e) = splice_inst %.loc7_3.35 [template = %.loc7_3.37 (constants.%.869)]
// CHECK:STDOUT:   %.loc7_3.38: <instruction> = call_action (%.loc7_3.14), true [template]
// CHECK:STDOUT:   %.loc7_3.39: type = type_of_inst %.loc7_3.38 [template = %.loc7_3.39 (constants.%.a78)]
// CHECK:STDOUT:   %.loc7_3.40: @F.%.loc7_3.39 (%.a78) = splice_inst %.loc7_3.38 [template = %.loc7_3.40 (constants.%.3d1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc7_20.12 (%.53f) = var_storage %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_25: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_3.1: type = type_of_inst %.loc7_3.17 [template = %.loc7_3.18 (constants.%.48a6b8.1)]
// CHECK:STDOUT:     %.loc7_3.2: @F.%.loc7_3.18 (%.48a6b8.1) = splice_inst %.loc7_3.17 [template = %.loc7_3.19 (constants.%.88d2c8.1)]
// CHECK:STDOUT:     %.loc7_3.3: type = type_of_inst %.loc7_3.20 [template = %.loc7_3.21 (constants.%.ceb5a2.1)]
// CHECK:STDOUT:     %.loc7_3.4: @F.%.loc7_3.21 (%.ceb5a2.1) = splice_inst %.loc7_3.20 [template = %.loc7_3.22 (constants.%.45c659.1)]
// CHECK:STDOUT:     %.loc7_3.5: type = type_of_inst %.loc7_3.23 [template = %.loc7_3.24 (constants.%.40e)]
// CHECK:STDOUT:     %.loc7_3.6: @F.%.loc7_3.24 (%.40e) = splice_inst %.loc7_3.23 [template = %.loc7_3.25 (constants.%.fa1)]
// CHECK:STDOUT:     %.loc7_3.7: type = type_of_inst %.loc7_3.26 [template = %.loc7_3.27 (constants.%.1fa)]
// CHECK:STDOUT:     %.loc7_3.8: @F.%.loc7_3.27 (%.1fa) = splice_inst %.loc7_3.26 [template = %.loc7_3.28 (constants.%.e13)]
// CHECK:STDOUT:     %.loc7_3.9: @F.%.loc7_20.12 (%.53f) = converted %.loc7_25, %.loc7_3.8 [template = %.loc7_3.28 (constants.%.e13)]
// CHECK:STDOUT:     %.loc7_3.10: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.29 [template = %.loc7_3.30 (constants.%.6ec)]
// CHECK:STDOUT:     %.loc7_3.11: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.31 [template = %.loc7_3.32 (constants.%.817)]
// CHECK:STDOUT:     assign %v.var, %.loc7_3.11
// CHECK:STDOUT:     %.loc7_20.1: type = splice_block %.loc7_20.5 [template = %.loc7_20.12 (constants.%.53f)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_20.2: type = splice_inst %.loc7_20.6 [template = %.loc7_20.7 (constants.%.eec)]
// CHECK:STDOUT:       %.loc7_20.3: type = type_of_inst %.loc7_20.8 [template = %.loc7_20.9 (constants.%.b01)]
// CHECK:STDOUT:       %.loc7_20.4: @F.%.loc7_20.9 (%.b01) = splice_inst %.loc7_20.8 [template = %.loc7_20.10 (constants.%.195)]
// CHECK:STDOUT:       %.loc7_20.5: type = splice_inst %.loc7_20.11 [template = %.loc7_20.12 (constants.%.53f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc7_20.12 (%.53f) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type (%pattern_type.7d1) = ref_binding_pattern v [template = %v.patt.loc7_15.2 (constants.%v.patt.7b6)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type (%pattern_type.7d1) = var_pattern %v.patt.loc7_15.1 [template = %v.var_patt.loc7_3.2 (constants.%v.var_patt.067)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_3.12: @F.%.loc7_20.12 (%.53f) = splice_inst %.loc7_3.33 [template = %.loc7_3.34 (constants.%.14f)]
// CHECK:STDOUT:     %.loc7_3.13: type = type_of_inst %.loc7_3.35 [template = %.loc7_3.36 (constants.%.c4e)]
// CHECK:STDOUT:     %.loc7_3.14: @F.%.loc7_3.36 (%.c4e) = splice_inst %.loc7_3.35 [template = %.loc7_3.37 (constants.%.869)]
// CHECK:STDOUT:     %.loc7_3.15: type = type_of_inst %.loc7_3.38 [template = %.loc7_3.39 (constants.%.a78)]
// CHECK:STDOUT:     %.loc7_3.16: @F.%.loc7_3.39 (%.a78) = splice_inst %.loc7_3.38 [template = %.loc7_3.40 (constants.%.3d1)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %C.d8efcc.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_20.6 => constants.%inst.specific_inst.c72
// CHECK:STDOUT:   %.loc7_20.7 => constants.%empty_struct_type
// CHECK:STDOUT:   %.loc7_20.8 => constants.%inst.class_type
// CHECK:STDOUT:   %.loc7_20.9 => type
// CHECK:STDOUT:   %.loc7_20.10 => constants.%C.d8efcc.1
// CHECK:STDOUT:   %.loc7_20.11 => constants.%inst.splice_block.1b2
// CHECK:STDOUT:   %.loc7_20.12 => constants.%C.d8efcc.1
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2fd
// CHECK:STDOUT:   %v.patt.loc7_15.2 => constants.%v.patt.0e7
// CHECK:STDOUT:   %v.var_patt.loc7_3.2 => constants.%v.var_patt.259
// CHECK:STDOUT:   %.loc7_3.17 => constants.%.0a6
// CHECK:STDOUT:   %.loc7_3.18 => constants.%.48a6b8.2
// CHECK:STDOUT:   %.loc7_3.19 => constants.%.88d2c8.2
// CHECK:STDOUT:   %.loc7_3.20 => constants.%.53c
// CHECK:STDOUT:   %.loc7_3.21 => constants.%.ceb5a2.2
// CHECK:STDOUT:   %.loc7_3.22 => constants.%.45c659.2
// CHECK:STDOUT:   %.loc7_3.23 => constants.%.1e2
// CHECK:STDOUT:   %.loc7_3.24 => constants.%.4f9
// CHECK:STDOUT:   %.loc7_3.25 => constants.%.bc1
// CHECK:STDOUT:   %.loc7_3.26 => constants.%.1ec
// CHECK:STDOUT:   %.loc7_3.27 => constants.%.65e
// CHECK:STDOUT:   %.loc7_3.28 => constants.%.fc2
// CHECK:STDOUT:   %.loc7_3.29 => constants.%.e48
// CHECK:STDOUT:   %.loc7_3.30 => constants.%.118
// CHECK:STDOUT:   %.loc7_3.31 => constants.%.663
// CHECK:STDOUT:   %.loc7_3.32 => constants.%.e51
// CHECK:STDOUT:   %.loc7_3.33 => constants.%inst.specific_inst.2e8
// CHECK:STDOUT:   %.loc7_3.34 => invalid
// CHECK:STDOUT:   %.loc7_3.35 => constants.%inst.splice_block.5e8
// CHECK:STDOUT:   %.loc7_3.36 => <bound method>
// CHECK:STDOUT:   %.loc7_3.37 => invalid
// CHECK:STDOUT:   %.loc7_3.38 => constants.%inst.call
// CHECK:STDOUT:   %.loc7_3.39 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc7_3.40 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

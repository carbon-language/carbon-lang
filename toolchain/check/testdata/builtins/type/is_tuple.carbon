// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/type/is_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/type/is_tuple.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

fn TypeIsTuple(t: type) -> bool = "type.is_tuple";

class ConstantTest(b:! bool) {}

fn True() -> ConstantTest(true);
fn False() -> ConstantTest(false);

fn F() {
  //@dump-sem-ir-begin
  var a: ConstantTest(TypeIsTuple(())) = True();
  var b: ConstantTest(TypeIsTuple({})) = False();
  var c: ConstantTest(TypeIsTuple(bool)) = False();
  //@dump-sem-ir-end
}

// --- fail_runtime.carbon
library "[[@TEST_NAME]]";

fn TypeIsTuple(t: type) -> bool = "type.is_tuple";

fn F(t: type) -> bool {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_runtime.carbon:[[@LINE+7]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return TypeIsTuple(t);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime.carbon:[[@LINE-7]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn TypeIsTuple(t: type) -> bool = "type.is_tuple";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TypeIsTuple(t);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %TypeIsTuple.type: type = fn_type @TypeIsTuple [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TypeIsTuple: %TypeIsTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstantTest.type: type = generic_class_type @ConstantTest [concrete]
// CHECK:STDOUT:   %ConstantTest.generic: %ConstantTest.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ConstantTest.97c: type = class_type @ConstantTest, @ConstantTest(%true) [concrete]
// CHECK:STDOUT:   %pattern_type.7d8: type = pattern_type %ConstantTest.97c [concrete]
// CHECK:STDOUT:   %True.type: type = fn_type @True [concrete]
// CHECK:STDOUT:   %True: %True.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %ConstantTest.033: type = class_type @ConstantTest, @ConstantTest(%false) [concrete]
// CHECK:STDOUT:   %pattern_type.415: type = pattern_type %ConstantTest.033 [concrete]
// CHECK:STDOUT:   %False.type: type = fn_type @False [concrete]
// CHECK:STDOUT:   %False: %False.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.type.56b: type = fn_type @ConstantTest.as.Destroy.impl.Op, @ConstantTest.as.Destroy.impl(%false) [concrete]
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.fcb: %ConstantTest.as.Destroy.impl.Op.type.56b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5c6: type = ptr_type %ConstantTest.033 [concrete]
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.type.cb4: type = fn_type @ConstantTest.as.Destroy.impl.Op, @ConstantTest.as.Destroy.impl(%true) [concrete]
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.db7: %ConstantTest.as.Destroy.impl.Op.type.cb4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e83: type = ptr_type %ConstantTest.97c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7d8 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7d8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ConstantTest.97c = var %a.var_patt
// CHECK:STDOUT:   %True.ref: %True.type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   %.loc12_3: ref %ConstantTest.97c = splice_block %a.var {}
// CHECK:STDOUT:   %True.call: init %ConstantTest.97c = call %True.ref() to %.loc12_3
// CHECK:STDOUT:   assign %a.var, %True.call
// CHECK:STDOUT:   %.loc12_38.1: type = splice_block %ConstantTest.loc12 [concrete = constants.%ConstantTest.97c] {
// CHECK:STDOUT:     %ConstantTest.ref.loc12: %ConstantTest.type = name_ref ConstantTest, file.%ConstantTest.decl [concrete = constants.%ConstantTest.generic]
// CHECK:STDOUT:     %TypeIsTuple.ref.loc12: %TypeIsTuple.type = name_ref TypeIsTuple, file.%TypeIsTuple.decl [concrete = constants.%TypeIsTuple]
// CHECK:STDOUT:     %.loc12_36.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_36.2: type = converted %.loc12_36.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %TypeIsTuple.call.loc12: init bool = call %TypeIsTuple.ref.loc12(%.loc12_36.2) [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_38.2: bool = value_of_initializer %TypeIsTuple.call.loc12 [concrete = constants.%true]
// CHECK:STDOUT:     %.loc12_38.3: bool = converted %TypeIsTuple.call.loc12, %.loc12_38.2 [concrete = constants.%true]
// CHECK:STDOUT:     %ConstantTest.loc12: type = class_type @ConstantTest, @ConstantTest(constants.%true) [concrete = constants.%ConstantTest.97c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ConstantTest.97c = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.415 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.415 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %ConstantTest.033 = var %b.var_patt
// CHECK:STDOUT:   %False.ref.loc13: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc13_3: ref %ConstantTest.033 = splice_block %b.var {}
// CHECK:STDOUT:   %False.call.loc13: init %ConstantTest.033 = call %False.ref.loc13() to %.loc13_3
// CHECK:STDOUT:   assign %b.var, %False.call.loc13
// CHECK:STDOUT:   %.loc13_38.1: type = splice_block %ConstantTest.loc13 [concrete = constants.%ConstantTest.033] {
// CHECK:STDOUT:     %ConstantTest.ref.loc13: %ConstantTest.type = name_ref ConstantTest, file.%ConstantTest.decl [concrete = constants.%ConstantTest.generic]
// CHECK:STDOUT:     %TypeIsTuple.ref.loc13: %TypeIsTuple.type = name_ref TypeIsTuple, file.%TypeIsTuple.decl [concrete = constants.%TypeIsTuple]
// CHECK:STDOUT:     %.loc13_36.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_36.2: type = converted %.loc13_36.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %TypeIsTuple.call.loc13: init bool = call %TypeIsTuple.ref.loc13(%.loc13_36.2) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc13_38.2: bool = value_of_initializer %TypeIsTuple.call.loc13 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc13_38.3: bool = converted %TypeIsTuple.call.loc13, %.loc13_38.2 [concrete = constants.%false]
// CHECK:STDOUT:     %ConstantTest.loc13: type = class_type @ConstantTest, @ConstantTest(constants.%false) [concrete = constants.%ConstantTest.033]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ConstantTest.033 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.415 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.415 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %ConstantTest.033 = var %c.var_patt
// CHECK:STDOUT:   %False.ref.loc14: %False.type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   %.loc14_3: ref %ConstantTest.033 = splice_block %c.var {}
// CHECK:STDOUT:   %False.call.loc14: init %ConstantTest.033 = call %False.ref.loc14() to %.loc14_3
// CHECK:STDOUT:   assign %c.var, %False.call.loc14
// CHECK:STDOUT:   %.loc14_40.1: type = splice_block %ConstantTest.loc14 [concrete = constants.%ConstantTest.033] {
// CHECK:STDOUT:     %ConstantTest.ref.loc14: %ConstantTest.type = name_ref ConstantTest, file.%ConstantTest.decl [concrete = constants.%ConstantTest.generic]
// CHECK:STDOUT:     %TypeIsTuple.ref.loc14: %TypeIsTuple.type = name_ref TypeIsTuple, file.%TypeIsTuple.decl [concrete = constants.%TypeIsTuple]
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_35.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc14_35.2: type = converted %Bool.call, %.loc14_35.1 [concrete = bool]
// CHECK:STDOUT:     %TypeIsTuple.call.loc14: init bool = call %TypeIsTuple.ref.loc14(%.loc14_35.2) [concrete = constants.%false]
// CHECK:STDOUT:     %.loc14_40.2: bool = value_of_initializer %TypeIsTuple.call.loc14 [concrete = constants.%false]
// CHECK:STDOUT:     %.loc14_40.3: bool = converted %TypeIsTuple.call.loc14, %.loc14_40.2 [concrete = constants.%false]
// CHECK:STDOUT:     %ConstantTest.loc14: type = class_type @ConstantTest, @ConstantTest(constants.%false) [concrete = constants.%ConstantTest.033]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %ConstantTest.033 = bind_name c, %c.var
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc14_3.1: <bound method> = bound_method %.loc14_3, constants.%ConstantTest.as.Destroy.impl.Op.fcb
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14_3.1: <bound method> = bound_method %.loc14_3, %ConstantTest.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc14_3.1: %ptr.5c6 = addr_of %.loc14_3
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc14_3.1: init %empty_tuple.type = call %bound_method.loc14_3.1(%addr.loc14_3.1)
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc14_3.2: <bound method> = bound_method %c.var, constants.%ConstantTest.as.Destroy.impl.Op.fcb
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14_3.2: <bound method> = bound_method %c.var, %ConstantTest.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc14_3.2: %ptr.5c6 = addr_of %c.var
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc14_3.2: init %empty_tuple.type = call %bound_method.loc14_3.2(%addr.loc14_3.2)
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc13_3.1: <bound method> = bound_method %.loc13_3, constants.%ConstantTest.as.Destroy.impl.Op.fcb
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13_3.1: <bound method> = bound_method %.loc13_3, %ConstantTest.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc13_3.1: %ptr.5c6 = addr_of %.loc13_3
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc13_3.1: init %empty_tuple.type = call %bound_method.loc13_3.1(%addr.loc13_3.1)
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc13_3.2: <bound method> = bound_method %b.var, constants.%ConstantTest.as.Destroy.impl.Op.fcb
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13_3.2: <bound method> = bound_method %b.var, %ConstantTest.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc13_3.2: %ptr.5c6 = addr_of %b.var
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc13_3.2: init %empty_tuple.type = call %bound_method.loc13_3.2(%addr.loc13_3.2)
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc12_3.1: <bound method> = bound_method %.loc12_3, constants.%ConstantTest.as.Destroy.impl.Op.db7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_3.1: <bound method> = bound_method %.loc12_3, %ConstantTest.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc12_3.1: %ptr.e83 = addr_of %.loc12_3
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc12_3.1: init %empty_tuple.type = call %bound_method.loc12_3.1(%addr.loc12_3.1)
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.bound.loc12_3.2: <bound method> = bound_method %a.var, constants.%ConstantTest.as.Destroy.impl.Op.db7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_3.2: <bound method> = bound_method %a.var, %ConstantTest.as.Destroy.impl.Op.specific_fn.6
// CHECK:STDOUT:   %addr.loc12_3.2: %ptr.e83 = addr_of %a.var
// CHECK:STDOUT:   %ConstantTest.as.Destroy.impl.Op.call.loc12_3.2: init %empty_tuple.type = call %bound_method.loc12_3.2(%addr.loc12_3.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_runtime.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TypeIsTuple.type: type = fn_type @TypeIsTuple [concrete]
// CHECK:STDOUT:   %TypeIsTuple: %TypeIsTuple.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%t.param: type) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TypeIsTuple.ref: %TypeIsTuple.type = name_ref TypeIsTuple, file.%TypeIsTuple.decl [concrete = constants.%TypeIsTuple]
// CHECK:STDOUT:   %t.ref: type = name_ref t, %t
// CHECK:STDOUT:   %TypeIsTuple.call: init bool = call %TypeIsTuple.ref(%t.ref)
// CHECK:STDOUT:   %.loc14_24.1: bool = value_of_initializer %TypeIsTuple.call
// CHECK:STDOUT:   %.loc14_24.2: bool = converted %TypeIsTuple.call, %.loc14_24.1
// CHECK:STDOUT:   return %.loc14_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

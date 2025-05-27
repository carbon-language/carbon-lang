// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/init.carbon

// --- init.carbon
library "[[@TEST_NAME]]";

class Class {
  var n: ();
  var next: Class*;
}

fn Make(n: (), next: Class*) -> Class {
  //@dump-sem-ir-begin
  return {.n = n, .next = next};
  //@dump-sem-ir-end
}

fn MakeReorder(n: (), next: Class*) -> Class {
  //@dump-sem-ir-begin
  return {.next = next, .n = n};
  //@dump-sem-ir-end
}

// --- init_as.carbon
library "[[@TEST_NAME]]";

class Class {
  var a: ();
  var b: ();
}

fn F() -> () {
  //@dump-sem-ir-begin
  return ({.a = (), .b = ()} as Class).a;
  //@dump-sem-ir-end
}

// --- init_nested.carbon
library "[[@TEST_NAME]]";

class Inner {
  var a: ();
  var b: ();
}

fn MakeInner() -> Inner;

class Outer {
  var c: Inner;
  var d: Inner;
}

fn MakeOuter() -> Outer {
  //@dump-sem-ir-begin
  return {.c = MakeInner(), .d = MakeInner()};
  //@dump-sem-ir-end
}

// --- fail_bad_init.carbon
library "[[@TEST_NAME]]";

class Class {
  var a: ();
  var b: ();
}

fn F() {
  // CHECK:STDERR: fail_bad_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 1 field [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = ()} as Class;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  {.a = ()} as Class;
  // CHECK:STDERR: fail_bad_init.carbon:[[@LINE+4]]:3: error: missing value for field `b` in struct initialization [StructInitMissingFieldInLiteral]
  // CHECK:STDERR:   {.a = (), .c = ()} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = (), .c = ()} as Class;
  // CHECK:STDERR: fail_bad_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 3 fields [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = (), .b = (), .c = ()} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = (), .b = (), .c = ()} as Class;
}

// --- fail_todo_init_as_inplace.carbon
library "[[@TEST_NAME]]";

class Class {
  var a: ();
  var b: ();
}

fn G(p: Class*);

fn F() {
  // TODO: This case should presumably work: `{...} as Class` should be an
  // initializing expression, not a value expression.
  //
  // CHECK:STDERR: fail_todo_init_as_inplace.carbon:[[@LINE+4]]:18: error: cannot copy value of type `Class` [CopyOfUncopyableType]
  // CHECK:STDERR:   var c: Class = {.a = (), .b = ()} as Class;
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var c: Class = {.a = (), .b = ()} as Class;
  G(&c);
}

// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %empty_tuple.type, .next: %ptr.e71} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.next.n: type = struct_type {.next: %ptr.e71, .n: %empty_tuple.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param: %empty_tuple.type, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %.loc10_31.1: %struct_type.n.next = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %.loc10_31.2: ref %empty_tuple.type = class_element_access %return, element0
// CHECK:STDOUT:   %.loc10_16: init %empty_tuple.type = tuple_init () to %.loc10_31.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_31.3: init %empty_tuple.type = converted %n.ref, %.loc10_16 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_31.4: ref %ptr.e71 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc10_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc10_31.4
// CHECK:STDOUT:   %.loc10_31.6: init %Class = class_init (%.loc10_31.3, %.loc10_31.5), %return
// CHECK:STDOUT:   %.loc10_32: init %Class = converted %.loc10_31.1, %.loc10_31.6
// CHECK:STDOUT:   return %.loc10_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param: %empty_tuple.type, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %.loc16_31.1: %struct_type.next.n = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %.loc16_31.2: ref %empty_tuple.type = class_element_access %return, element1
// CHECK:STDOUT:   %.loc16_30: init %empty_tuple.type = tuple_init () to %.loc16_31.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_31.3: init %empty_tuple.type = converted %n.ref, %.loc16_30 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_31.4: ref %ptr.e71 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc16_31.5: init %ptr.e71 = initialize_from %next.ref to %.loc16_31.4
// CHECK:STDOUT:   %.loc16_31.6: init %Class = class_init (%.loc16_31.3, %.loc16_31.5), %return
// CHECK:STDOUT:   %.loc16_32: init %Class = converted %.loc16_31.1, %.loc16_31.6
// CHECK:STDOUT:   return %.loc16_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_28.1: %struct_type.a.b = struct_literal (%.loc10_18.1, %.loc10_27.1)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc10_28.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc10_28.3: ref %empty_tuple.type = class_element_access %.loc10_28.2, element0
// CHECK:STDOUT:   %.loc10_18.2: init %empty_tuple.type = tuple_init () to %.loc10_28.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_28.4: init %empty_tuple.type = converted %.loc10_18.1, %.loc10_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_28.5: ref %empty_tuple.type = class_element_access %.loc10_28.2, element1
// CHECK:STDOUT:   %.loc10_27.2: init %empty_tuple.type = tuple_init () to %.loc10_28.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_28.6: init %empty_tuple.type = converted %.loc10_27.1, %.loc10_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_28.7: init %Class = class_init (%.loc10_28.4, %.loc10_28.6), %.loc10_28.2 [concrete = constants.%Class.val]
// CHECK:STDOUT:   %.loc10_28.8: ref %Class = temporary %.loc10_28.2, %.loc10_28.7
// CHECK:STDOUT:   %.loc10_30: ref %Class = converted %.loc10_28.1, %.loc10_28.8
// CHECK:STDOUT:   %a.ref: %Class.elem = name_ref a, @Class.%.loc4_8 [concrete = @Class.%.loc4_8]
// CHECK:STDOUT:   %.loc10_39: ref %empty_tuple.type = class_element_access %.loc10_30, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_41: %empty_tuple.type = converted %.loc10_39, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc10_41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %MakeInner.type: type = fn_type @MakeInner [concrete]
// CHECK:STDOUT:   %MakeInner: %MakeInner.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %struct_type.c.d.dce: type = struct_type {.c: %Inner, .d: %Inner} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeOuter() -> %return.param: %Outer {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeInner.ref.loc17_16: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [concrete = constants.%MakeInner]
// CHECK:STDOUT:   %.loc17_45.1: ref %Inner = class_element_access %return, element0
// CHECK:STDOUT:   %MakeInner.call.loc17_26: init %Inner = call %MakeInner.ref.loc17_16() to %.loc17_45.1
// CHECK:STDOUT:   %MakeInner.ref.loc17_34: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [concrete = constants.%MakeInner]
// CHECK:STDOUT:   %.loc17_45.2: ref %Inner = class_element_access %return, element1
// CHECK:STDOUT:   %MakeInner.call.loc17_44: init %Inner = call %MakeInner.ref.loc17_34() to %.loc17_45.2
// CHECK:STDOUT:   %.loc17_45.3: %struct_type.c.d.dce = struct_literal (%MakeInner.call.loc17_26, %MakeInner.call.loc17_44)
// CHECK:STDOUT:   %.loc17_45.4: init %Outer = class_init (%MakeInner.call.loc17_26, %MakeInner.call.loc17_44), %return
// CHECK:STDOUT:   %.loc17_46: init %Outer = converted %.loc17_45.3, %.loc17_45.4
// CHECK:STDOUT:   return %.loc17_46 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/fields.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/fields.carbon

// --- field_access.carbon
library "[[@TEST_NAME]]";

class Class {
  var j: ();
  var k: ();
}

fn Run() {
  var c: Class;
  //@dump-sem-ir-begin
  c.j = ();
  c.k = ();
  var cj: () = c.j;
  var ck: () = c.k;
  //@dump-sem-ir-end
}

// --- field_access_in_value.carbon
library "[[@TEST_NAME]]";

class Class {
  var j: ();
  var k: ();
}

fn Test() {
  var cv: Class;
  cv.j = ();
  cv.k = ();
  let c: Class = cv;
  //@dump-sem-ir-begin
  var cj: () = c.j;
  var ck: () = c.k;
  //@dump-sem-ir-end
}

// --- fail_todo_field_initializer.carbon
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_todo_field_initializer.carbon:[[@LINE+4]]:3: error: semantics TODO: `Field initializer` [SemanticsTodo]
  // CHECK:STDERR:   var field: () = 0;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var field: () = 0;
}

// --- fail_field_modifiers.carbon
library "[[@TEST_NAME]]";

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   default var j: ();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default var j: ();

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed on `var` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   final var k: ();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final var k: ();

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default let l: () = ();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let l: () = ();

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   final let m: () = ();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final let m: () = ();
}

// --- fail_unbound_field.carbon
library "[[@TEST_NAME]]";

class Class {
  var field: ();
  fn F() -> () {
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+4]]:12: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     return field;
    // CHECK:STDERR:            ^~~~~
    // CHECK:STDERR:
    return field;
  }
}

fn G() -> () {
  // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+4]]:10: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   return Class.field;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  return Class.field;
}

// CHECK:STDOUT: --- field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c.ref.loc11: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc11: %Class.elem = name_ref j, @Class.%.loc4_8 [concrete = @Class.%.loc4_8]
// CHECK:STDOUT:   %.loc11_4: ref %empty_tuple.type = class_element_access %c.ref.loc11, element0
// CHECK:STDOUT:   %.loc11_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_10.2: init %empty_tuple.type = tuple_init () to %.loc11_4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_7: init %empty_tuple.type = converted %.loc11_10.1, %.loc11_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc11_4, %.loc11_7
// CHECK:STDOUT:   %c.ref.loc12: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc12: %Class.elem = name_ref k, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:   %.loc12_4: ref %empty_tuple.type = class_element_access %c.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_10.2: init %empty_tuple.type = tuple_init () to %.loc12_4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_7: init %empty_tuple.type = converted %.loc12_10.1, %.loc12_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc12_4, %.loc12_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cj.patt: %pattern_type.cb1 = binding_pattern cj [concrete]
// CHECK:STDOUT:     %cj.var_patt: %pattern_type.cb1 = var_pattern %cj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj.var: ref %empty_tuple.type = var %cj.var_patt
// CHECK:STDOUT:   %c.ref.loc13: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc13: %Class.elem = name_ref j, @Class.%.loc4_8 [concrete = @Class.%.loc4_8]
// CHECK:STDOUT:   %.loc13_17.1: ref %empty_tuple.type = class_element_access %c.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_17.2: init %empty_tuple.type = tuple_init () to %cj.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_3: init %empty_tuple.type = converted %.loc13_17.1, %.loc13_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %cj.var, %.loc13_3
// CHECK:STDOUT:   %.loc13_12.1: type = splice_block %.loc13_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc13_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_12.3: type = converted %.loc13_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj: ref %empty_tuple.type = bind_name cj, %cj.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ck.patt: %pattern_type.cb1 = binding_pattern ck [concrete]
// CHECK:STDOUT:     %ck.var_patt: %pattern_type.cb1 = var_pattern %ck.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck.var: ref %empty_tuple.type = var %ck.var_patt
// CHECK:STDOUT:   %c.ref.loc14: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc14: %Class.elem = name_ref k, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:   %.loc14_17.1: ref %empty_tuple.type = class_element_access %c.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_17.2: init %empty_tuple.type = tuple_init () to %ck.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_3: init %empty_tuple.type = converted %.loc14_17.1, %.loc14_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %ck.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_12.1: type = splice_block %.loc14_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc14_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_12.3: type = converted %.loc14_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck: ref %empty_tuple.type = bind_name ck, %ck.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- field_access_in_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cj.patt: %pattern_type.cb1 = binding_pattern cj [concrete]
// CHECK:STDOUT:     %cj.var_patt: %pattern_type.cb1 = var_pattern %cj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj.var: ref %empty_tuple.type = var %cj.var_patt
// CHECK:STDOUT:   %c.ref.loc14: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc14: %Class.elem = name_ref j, @Class.%.loc4_8 [concrete = @Class.%.loc4_8]
// CHECK:STDOUT:   %.loc14_17.1: ref %empty_tuple.type = class_element_access %c.ref.loc14, element0
// CHECK:STDOUT:   %.loc14_17.2: init %empty_tuple.type = tuple_init () to %cj.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_3: init %empty_tuple.type = converted %.loc14_17.1, %.loc14_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %cj.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_12.1: type = splice_block %.loc14_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc14_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_12.3: type = converted %.loc14_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj: ref %empty_tuple.type = bind_name cj, %cj.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ck.patt: %pattern_type.cb1 = binding_pattern ck [concrete]
// CHECK:STDOUT:     %ck.var_patt: %pattern_type.cb1 = var_pattern %ck.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck.var: ref %empty_tuple.type = var %ck.var_patt
// CHECK:STDOUT:   %c.ref.loc15: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc15: %Class.elem = name_ref k, @Class.%.loc5_8 [concrete = @Class.%.loc5_8]
// CHECK:STDOUT:   %.loc15_17.1: ref %empty_tuple.type = class_element_access %c.ref.loc15, element1
// CHECK:STDOUT:   %.loc15_17.2: init %empty_tuple.type = tuple_init () to %ck.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_3: init %empty_tuple.type = converted %.loc15_17.1, %.loc15_17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %ck.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_12.1: type = splice_block %.loc15_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc15_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.3: type = converted %.loc15_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck: ref %empty_tuple.type = bind_name ck, %ck.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

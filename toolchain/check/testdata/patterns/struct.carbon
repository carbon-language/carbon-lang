// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// ^ARGS: -v compile %s
// ^EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/struct.carbon

// --- empty.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {} = {};
//@dump-sem-ir-end

// --- literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = b: ()} = {.a = ()};
//@dump-sem-ir-end

// --- literal_two_fields.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = b: (), .c = d: ()} = {.a = (), .c = ()};
//@dump-sem-ir-end

// --- literal_shorthand.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: ()} = {.a = ()};
//@dump-sem-ir-end

// --- literal_shorthand_two_fields.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: (), b: ()} = {.a = (), .b = ()};
//@dump-sem-ir-end

// --- literal_mixed_syntax.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = b: (), c: ()} = {.a = (), .c = ()};
//@dump-sem-ir-end

// --- nested.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = x: (), .b = {.i = m:(), .j = n:()}} = {.a = (), .b = {.i = (), .j = ()}};
//@dump-sem-ir-end

// --- init_expr.carbon

library "[[@TEST_NAME]]";

fn F() -> {.a: (), .b: ()};

//@dump-sem-ir-begin
let {.a = x: (), .b = y: ()} = F();
//@dump-sem-ir-end

// --- init_expr_shorthand.carbon

library "[[@TEST_NAME]]";

fn F() -> {.a: (), .b: ()};

//@dump-sem-ir-begin
let {a: (), b: ()} = F();
//@dump-sem-ir-end

// --- ref_expr.carbon

library "[[@TEST_NAME]]";

var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

//@dump-sem-ir-begin
let {.a = x: i32, .b = y: i32} = s;
//@dump-sem-ir-end

// --- ref_expr_shorthand.carbon

library "[[@TEST_NAME]]";

var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

//@dump-sem-ir-begin
let {a: i32, b: i32} = s;
//@dump-sem-ir-end

// --- fail_todo_underscore.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_underscore.carbon:[[@LINE+4]]:5: error: semantics TODO: `Struct pattern underscore field` [SemanticsTodo]
// CHECK:STDERR: let {.a = x: (), _} = {.a = (), .b = ()};
// CHECK:STDERR:     ^~~~~~~~~~~~~~~
// CHECK:STDERR:
let {.a = x: (), _} = {.a = (), .b = ()};
//@dump-sem-ir-end

// --- fail_no_matching_field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_no_matching_field.carbon:[[@LINE+4]]:20: error: type `{.b: ()}` does not have a member `a` [StructPatternNoMatchingField]
// CHECK:STDERR: let {.a = x: ()} = {.b = ()};
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = {.b = ()};
//@dump-sem-ir-end


// --- fail_unhandled_field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_unhandled_field.carbon:[[@LINE+4]]:20: error: scrutinee field `.b` not matched by provided pattern [StructPatternUnhandledField]
// CHECK:STDERR: let {.a = x: ()} = {.a = (), .b = ()};
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = {.a = (), .b = ()};
//@dump-sem-ir-end

// --- fail_duplicate_field_name.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate_field_name.carbon:[[@LINE+7]]:19: error: duplicated field name `a` in struct pattern [StructPatternNameDuplicate]
// CHECK:STDERR: let {.a = b: (), .a = c: ()} = {.a = ()};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_field_name.carbon:[[@LINE+4]]:7: note: field with the same name here [StructPatternNamePrevious]
// CHECK:STDERR: let {.a = b: (), .a = c: ()} = {.a = ()};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.a = b: (), .a = c: ()} = {.a = ()};
//@dump-sem-ir-end

// --- fail_duplicate_field_name_shorthand.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+14]]:13: error: duplicate name `a` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+11]]:6: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:      ^~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+7]]:13: error: duplicated field name `a` in struct pattern [StructPatternNameDuplicate]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+4]]:6: note: field with the same name here [StructPatternNamePrevious]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {a: (), a: ()} = {.a = ()};
//@dump-sem-ir-end

// --- fail_duplicate_field_name_mixed_syntax.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate_field_name_mixed_syntax.carbon:[[@LINE+7]]:18: error: duplicated field name `a` in struct pattern [StructPatternNameDuplicate]
// CHECK:STDERR: let {.a = b: (), a: ()} = {.a = ()};
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_duplicate_field_name_mixed_syntax.carbon:[[@LINE+4]]:7: note: field with the same name here [StructPatternNamePrevious]
// CHECK:STDERR: let {.a = b: (), a: ()} = {.a = ()};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.a = b: (), a: ()} = {.a = ()};
//@dump-sem-ir-end

// --- fail_in_class_var.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C{
    // CHECK:STDERR: fail_in_class_var.carbon:[[@LINE+4]]:9: error: found struct pattern in class member var declaration [FieldWithStructPattern]
    // CHECK:STDERR:     var {.a = x: ()};
    // CHECK:STDERR:         ^~~~~~~~~~~~
    // CHECK:STDERR:
    var {.a = x: ()};
}
//@dump-sem-ir-end

// --- fail_in_class_let.carbon

library "[[@TEST_NAME]]";
//@dump-sem-ir-begin
class C{
    // CHECK:STDERR: fail_in_class_let.carbon:[[@LINE+4]]:9: error: found struct pattern in class member let declaration [FieldWithStructPattern]
    // CHECK:STDERR:     let {.a = x: ()} = {.a = ()};
    // CHECK:STDERR:         ^~~~~~~~~~~~
    // CHECK:STDERR:
    let {.a = x: ()} = {.a = ()};
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.931: %pattern_type = struct_pattern () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5: %pattern_type = struct_pattern () [concrete = constants.%.931]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.140: %pattern_type.e85 = struct_pattern (%b.patt) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_28.1: %empty_tuple.type = struct_access @__global_init.%.loc5_28, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.2: %empty_tuple.type = converted %.loc5_28.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_16: %pattern_type.e85 = struct_pattern (%b.patt) [concrete = constants.%.140]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_27: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28: %struct_type.a = struct_literal (%.loc5_27) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_two_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.cb1 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %struct_type.a.c: type = struct_type {.a: %empty_tuple.type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.f10: type = pattern_type %struct_type.a.c [concrete]
// CHECK:STDOUT:   %.ab2: %pattern_type.f10 = struct_pattern (%b.patt, %d.patt) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.c = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_49.1: %empty_tuple.type = struct_access @__global_init.%.loc5_49, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.2: %empty_tuple.type = struct_access @__global_init.%.loc5_49, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_49.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.3: %empty_tuple.type = converted %.loc5_49.1, %empty_tuple.loc5_49.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_49.3
// CHECK:STDOUT:   %empty_tuple.loc5_49.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.4: %empty_tuple.type = converted %.loc5_49.2, %empty_tuple.loc5_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_27.1: type = splice_block %.loc5_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_27.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %.loc5_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: %empty_tuple.type = wrapper_binding d, %.loc5_49.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %.loc5_28: %pattern_type.f10 = struct_pattern (%b.patt, %d.patt) [concrete = constants.%.ab2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_39: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_48: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49: %struct_type.a.c = struct_literal (%.loc5_39, %.loc5_48) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.e73: %pattern_type.e85 = struct_pattern (%a.patt) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_23.1: %empty_tuple.type = struct_access @__global_init.%.loc5_23, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_23.2: %empty_tuple.type = converted %.loc5_23.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %.loc5_11: %pattern_type.e85 = struct_pattern (%a.patt) [concrete = constants.%.e73]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_22: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_23: %struct_type.a = struct_literal (%.loc5_22) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_shorthand_two_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %.c90: %pattern_type.d8f = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_39.1: %empty_tuple.type = struct_access @__global_init.%.loc5_39, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_39.2: %empty_tuple.type = struct_access @__global_init.%.loc5_39, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_39.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_39.3: %empty_tuple.type = converted %.loc5_39.1, %empty_tuple.loc5_39.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_39.3
// CHECK:STDOUT:   %empty_tuple.loc5_39.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_39.4: %empty_tuple.type = converted %.loc5_39.2, %empty_tuple.loc5_39.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_17.1: type = splice_block %.loc5_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_17.3: type = converted %.loc5_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_39.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_18: %pattern_type.d8f = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.c90]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_29: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_38: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_39: %struct_type.a.b = struct_literal (%.loc5_29, %.loc5_38) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_mixed_syntax.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.cb1 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %struct_type.a.c: type = struct_type {.a: %empty_tuple.type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.f10: type = pattern_type %struct_type.a.c [concrete]
// CHECK:STDOUT:   %.fcd: %pattern_type.f10 = struct_pattern (%b.patt, %c.patt) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.c = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_44.1: %empty_tuple.type = struct_access @__global_init.%.loc5_44, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.2: %empty_tuple.type = struct_access @__global_init.%.loc5_44, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_44.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.3: %empty_tuple.type = converted %.loc5_44.1, %empty_tuple.loc5_44.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_44.3
// CHECK:STDOUT:   %empty_tuple.loc5_44.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.4: %empty_tuple.type = converted %.loc5_44.2, %empty_tuple.loc5_44.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_22.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_22.3: type = converted %.loc5_22.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %empty_tuple.type = wrapper_binding c, %.loc5_44.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %.loc5_23: %pattern_type.f10 = struct_pattern (%b.patt, %c.patt) [concrete = constants.%.fcd]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_34: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_43: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44: %struct_type.a.c = struct_literal (%.loc5_34, %.loc5_43) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %m.patt: %pattern_type.cb1 = value_binding_pattern m [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.cb1 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %struct_type.i.j: type = struct_type {.i: %empty_tuple.type, .j: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.63f: type = pattern_type %struct_type.i.j [concrete]
// CHECK:STDOUT:   %.dfb: %pattern_type.63f = struct_pattern (%m.patt, %n.patt) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %struct_type.i.j} [concrete]
// CHECK:STDOUT:   %pattern_type.543: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %.e73: %pattern_type.543 = struct_pattern (%x.patt, %.dfb) [concrete]
// CHECK:STDOUT:   %struct.4fa: %struct_type.i.j = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %struct.b22: %struct_type.a.b = struct_value (%empty_tuple, %struct.4fa) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5_82.1: %empty_tuple.type = struct_access @__global_init.%.loc5_82, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.2: %struct_type.i.j = struct_access @__global_init.%.loc5_82, element1 [concrete = constants.%struct.4fa]
// CHECK:STDOUT:   %empty_tuple.loc5_82.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.3: %empty_tuple.type = converted %.loc5_82.1, %empty_tuple.loc5_82.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, %.loc5_82.3
// CHECK:STDOUT:   %.loc5_82.4: %empty_tuple.type = struct_access %.loc5_82.2, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_82.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.5: %empty_tuple.type = converted %.loc5_82.4, %empty_tuple.loc5_82.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.6: %empty_tuple.type = struct_access %.loc5_82.2, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_82.3: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.7: %empty_tuple.type = converted %.loc5_82.6, %empty_tuple.loc5_82.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.i.j = struct_value (%.loc5_82.5, %.loc5_82.7) [concrete = constants.%struct.4fa]
// CHECK:STDOUT:   %.loc5_44.1: %struct_type.i.j = converted %.loc5_82.2, %struct [concrete = constants.%struct.4fa]
// CHECK:STDOUT:   %.loc5_82.8: %empty_tuple.type = struct_access %.loc5_44.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_82.9: %empty_tuple.type = struct_access %.loc5_44.1, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_32.1: type = splice_block %.loc5_32.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_32.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_32.3: type = converted %.loc5_32.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: %empty_tuple.type = wrapper_binding m, %.loc5_82.8
// CHECK:STDOUT:   %.loc5_43.1: type = splice_block %.loc5_43.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_43.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_43.3: type = converted %.loc5_43.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %empty_tuple.type = wrapper_binding n, %.loc5_82.9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.cb1 = value_binding_pattern m [concrete = constants.%m.patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.cb1 = value_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     %.loc5_44.2: %pattern_type.63f = struct_pattern (%m.patt, %n.patt) [concrete = constants.%.dfb]
// CHECK:STDOUT:     %.loc5_45: %pattern_type.543 = struct_pattern (%x.patt, %.loc5_44.2) [concrete = constants.%.e73]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_56: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_71: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_80: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_81: %struct_type.i.j = struct_literal (%.loc5_71, %.loc5_80) [concrete = constants.%struct.4fa]
// CHECK:STDOUT:   %.loc5_82: %struct_type.a.b = struct_literal (%.loc5_56, %.loc5_81) [concrete = constants.%struct.b22]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %.9fc: %pattern_type.d8f = struct_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_34.1: ref %struct_type.a.b = temporary @__global_init.%.loc7, @__global_init.%F.call
// CHECK:STDOUT:   %.loc7_34.2: ref %empty_tuple.type = struct_access %.loc7_34.1, element0
// CHECK:STDOUT:   %.loc7_34.3: ref %empty_tuple.type = struct_access %.loc7_34.1, element1
// CHECK:STDOUT:   %empty_tuple.loc7_34.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_34.4: %empty_tuple.type = converted %.loc7_34.2, %empty_tuple.loc7_34.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_15.1: type = splice_block %.loc7_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_15.3: type = converted %.loc7_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, %.loc7_34.4
// CHECK:STDOUT:   %empty_tuple.loc7_34.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_34.5: %empty_tuple.type = converted %.loc7_34.3, %empty_tuple.loc7_34.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.1: type = splice_block %.loc7_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_27.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %.loc7_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: %empty_tuple.type = wrapper_binding y, %.loc7_34.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc7_28: %pattern_type.d8f = struct_pattern (%x.patt, %y.patt) [concrete = constants.%.9fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %struct_type.a.b = temporary_storage
// CHECK:STDOUT:   %F.call: init %struct_type.a.b to %.loc7 = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.c90: %pattern_type.d8f = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_24.1: ref %struct_type.a.b = temporary @__global_init.%.loc7, @__global_init.%F.call
// CHECK:STDOUT:   %.loc7_24.2: ref %empty_tuple.type = struct_access %.loc7_24.1, element0
// CHECK:STDOUT:   %.loc7_24.3: ref %empty_tuple.type = struct_access %.loc7_24.1, element1
// CHECK:STDOUT:   %empty_tuple.loc7_24.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.4: %empty_tuple.type = converted %.loc7_24.2, %empty_tuple.loc7_24.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_10.3: type = converted %.loc7_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc7_24.4
// CHECK:STDOUT:   %empty_tuple.loc7_24.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.5: %empty_tuple.type = converted %.loc7_24.3, %empty_tuple.loc7_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_17.1: type = splice_block %.loc7_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_17.3: type = converted %.loc7_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc7_24.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7_18: %pattern_type.d8f = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.c90]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7: ref %struct_type.a.b = temporary_storage
// CHECK:STDOUT:   %F.call: init %struct_type.a.b to %.loc7 = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.20e3: type = pattern_type %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %.9f1: ref %i32 = struct_access file.%s.var, element0 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.e6f: ref %i32 = struct_access file.%s.var, element1 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.6b6 = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %.6c2: %pattern_type.20e3 = struct_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_34.1: ref %i32 = struct_access @__global_init.%s.ref, element0 [concrete = constants.%.9f1]
// CHECK:STDOUT:   %.loc7_34.2: ref %i32 = struct_access @__global_init.%s.ref, element1 [concrete = constants.%.e6f]
// CHECK:STDOUT:   %.loc7_34.3: %i32 = acquire_value %.loc7_34.1
// CHECK:STDOUT:   %i32.loc7_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc7_34.3
// CHECK:STDOUT:   %.loc7_34.4: %i32 = acquire_value %.loc7_34.2
// CHECK:STDOUT:   %i32.loc7_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: %i32 = wrapper_binding y, %.loc7_34.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.6b6 = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc7_30: %pattern_type.20e3 = struct_pattern (%x.patt, %y.patt) [concrete = constants.%.6c2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %s.ref: ref %struct_type.a.b.b4c = name_ref s, file.%s [concrete = file.%s.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.b4c: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.20e3: type = pattern_type %struct_type.a.b.b4c [concrete]
// CHECK:STDOUT:   %.9f1: ref %i32 = struct_access file.%s.var, element0 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.e6f: ref %i32 = struct_access file.%s.var, element1 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.0f5: %pattern_type.20e3 = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_24.1: ref %i32 = struct_access @__global_init.%s.ref, element0 [concrete = constants.%.9f1]
// CHECK:STDOUT:   %.loc7_24.2: ref %i32 = struct_access @__global_init.%s.ref, element1 [concrete = constants.%.e6f]
// CHECK:STDOUT:   %.loc7_24.3: %i32 = acquire_value %.loc7_24.1
// CHECK:STDOUT:   %i32.loc7_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc7_24.3
// CHECK:STDOUT:   %.loc7_24.4: %i32 = acquire_value %.loc7_24.2
// CHECK:STDOUT:   %i32.loc7_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc7_24.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7_20: %pattern_type.20e3 = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.0f5]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %s.ref: ref %struct_type.a.b.b4c = name_ref s, file.%s [concrete = file.%s.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_underscore.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_matching_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.298: %pattern_type.e85 = struct_pattern (%x.patt) [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc9: %pattern_type.e85 = struct_pattern (%x.patt) [concrete = constants.%.298]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_27: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_28: %struct_type.b = struct_literal (%.loc9_27) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_unhandled_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.298: %pattern_type.e85 = struct_pattern (%x.patt) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc9_37.1: %empty_tuple.type = struct_access @__global_init.%.loc9_37, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_37.2: %empty_tuple.type = converted %.loc9_37.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_15.1: type = splice_block %.loc9_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_15.3: type = converted %.loc9_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, %.loc9_37.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc9_16: %pattern_type.e85 = struct_pattern (%x.patt) [concrete = constants.%.298]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_27: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_36: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_37: %struct_type.a.b = struct_literal (%.loc9_27, %.loc9_36) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_duplicate_field_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.patt: %pattern_type = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_39: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_40: %struct_type.a = struct_literal (%.loc12_39) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_duplicate_field_name_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt.loc19_7: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.patt.loc19_14: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_29: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_30: %struct_type.a = struct_literal (%.loc19_29) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_duplicate_field_name_mixed_syntax.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_34: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_35: %struct_type.a = struct_literal (%.loc12_34) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_class_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = <unexpected>.inst{{[0-9A-F]+}}.loc10_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_class_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = <unexpected>.inst{{[0-9A-F]+}}.loc9_16
// CHECK:STDOUT: }
// CHECK:STDOUT:

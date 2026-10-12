// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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
let {.a = b: i32, .c = d: bool} = {.a = 1, .c = true};
//@dump-sem-ir-end

// --- literal_shorthand.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: ()} = {.a = ()};
//@dump-sem-ir-end

// --- literal_shorthand_two_fields.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: i32, b: bool} = {.a = 1, .b = true};
//@dump-sem-ir-end

// --- literal_mixed_syntax.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = b: (), c: ()} = {.a = (), .c = ()};
//@dump-sem-ir-end

// --- nested.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = x: i32, .b = {.i = m: f32, .j = n: bool}} = {.a = 1, .b = {.i = 2.0, .j = true}};
//@dump-sem-ir-end

// --- init_expr.carbon

library "[[@TEST_NAME]]";

fn F() -> {.a: i32, .b: bool};

//@dump-sem-ir-begin
let {.a = x: i32, .b = y: bool} = F();
//@dump-sem-ir-end

// --- init_expr_shorthand.carbon

library "[[@TEST_NAME]]";

fn F() -> {.a: i32, .b: bool};

//@dump-sem-ir-begin
let {a: i32, b: bool} = F();
//@dump-sem-ir-end

// --- ref_expr.carbon

library "[[@TEST_NAME]]";

var s: {.a: i32, .b: bool} = {.a = 1, .b = true};

//@dump-sem-ir-begin
let {.a = x: i32, .b = y: bool} = s;
//@dump-sem-ir-end

// --- ref_expr_shorthand.carbon

library "[[@TEST_NAME]]";

var s: {.a: i32, .b: bool} = {.a = 1, .b = true};

//@dump-sem-ir-begin
let {a: i32, b: bool} = s;
//@dump-sem-ir-end

// --- unordered.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {.a = b: i32, .c = d: bool} = {.c = true, .a = 1};
//@dump-sem-ir-end

// --- unordered_shorthand.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: i32, b: bool} = {.b = true, .a = 1};
//@dump-sem-ir-end

// --- unordered_mixed_syntax.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let {a: i32, .b = c: bool} = {.b = true, .a = 1};
//@dump-sem-ir-end

// --- fail_todo_underscore.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_todo_underscore.carbon:[[@LINE+4]]:5: error: semantics TODO: `Struct pattern underscore field` [SemanticsTodo]
// CHECK:STDERR: let {.a = x: i32, _} = {.a = 1, .b = ()};
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let {.a = x: i32, _} = {.a = 1, .b = ()};
//@dump-sem-ir-end

// --- fail_no_matching_field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_no_matching_field.carbon:[[@LINE+4]]:20: error: struct `{.a: ()}` has no field named `b` [StructInitUnexpectedFieldInLiteral]
// CHECK:STDERR: let {.a = x: ()} = {.b = ()};
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = {.b = ()};
//@dump-sem-ir-end

// --- fail_no_matching_field_ref.carbon

library "[[@TEST_NAME]]";
var s: {.a: (), .b: ()} = {.a = (), .b = ()};
//@dump-sem-ir-begin
// CHECK:STDERR: fail_no_matching_field_ref.carbon:[[@LINE+4]]:20: error: cannot convert from struct type `{.a: (), .b: ()}` to `{.c: ()}`: unexpected field `a` in source type [StructInitUnexpectedFieldInConversion]
// CHECK:STDERR: let {.c = x: ()} = s;
// CHECK:STDERR:                    ^
// CHECK:STDERR:
let {.c = x: ()} = s;
//@dump-sem-ir-end


// --- fail_unhandled_field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_unhandled_field.carbon:[[@LINE+4]]:20: error: struct `{.a: ()}` has no field named `b` [StructInitUnexpectedFieldInLiteral]
// CHECK:STDERR: let {.a = x: ()} = {.a = (), .b = true};
// CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = {.a = (), .b = true};
//@dump-sem-ir-end

// --- fail_duplicate_field_name.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate_field_name.carbon:[[@LINE+7]]:19: error: duplicated field name `a` in struct pattern [StructNameDuplicate]
// CHECK:STDERR: let {.a = b: (), .a = c: ()} = {.a = ()};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_field_name.carbon:[[@LINE+4]]:7: note: field with the same name here [StructNamePrevious]
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
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+7]]:13: error: duplicated field name `a` in struct pattern [StructNameDuplicate]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_duplicate_field_name_shorthand.carbon:[[@LINE+4]]:6: note: field with the same name here [StructNamePrevious]
// CHECK:STDERR: let {a: (), a: ()} = {.a = ()};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {a: (), a: ()} = {.a = ()};
//@dump-sem-ir-end

// --- fail_duplicate_field_name_mixed_syntax.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate_field_name_mixed_syntax.carbon:[[@LINE+7]]:18: error: duplicated field name `a` in struct pattern [StructNameDuplicate]
// CHECK:STDERR: let {.a = b: (), a: ()} = {.a = ()};
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_duplicate_field_name_mixed_syntax.carbon:[[@LINE+4]]:7: note: field with the same name here [StructNamePrevious]
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

// --- fail_incorrect_field_type.carbon

library "[[@TEST_NAME]]";
//@dump-sem-ir-begin
// CHECK:STDERR: fail_incorrect_field_type.carbon:[[@LINE+7]]:20: error: cannot implicitly convert expression of type `{}` to `()` [ConversionFailure]
// CHECK:STDERR: let {.a = x: ()} = {.a = {}};
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR: fail_incorrect_field_type.carbon:[[@LINE+4]]:20: note: type `{}` does not implement interface `Core.ImplicitAs(())` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let {.a = x: ()} = {.a = {}};
// CHECK:STDERR:                    ^~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = {.a = {}};
//@dump-sem-ir-end

// --- fail_incorrect_scrutinee_type.carbon

library "[[@TEST_NAME]]";
//@dump-sem-ir-begin
// CHECK:STDERR: fail_incorrect_scrutinee_type.carbon:[[@LINE+7]]:5: error: cannot implicitly convert expression of type `((), ())` to `{.a: ()}` [ConversionFailure]
// CHECK:STDERR: let {.a = x: ()} = ((), ());
// CHECK:STDERR:     ^~~~~~~~~~~~
// CHECK:STDERR: fail_incorrect_scrutinee_type.carbon:[[@LINE+4]]:5: note: type `((), ())` does not implement interface `Core.ImplicitAs({.a: ()})` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let {.a = x: ()} = ((), ());
// CHECK:STDERR:     ^~~~~~~~~~~~
// CHECK:STDERR:
let {.a = x: ()} = ((), ());
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
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_6.1: %empty_struct_type = converted @__global_init.%.loc5, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_6.2: %pattern_type = struct_pattern () [concrete = constants.%.931]
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28: %empty_tuple.type = converted @__global_init.%.loc5_27, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%.loc5_28) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_16.1: %struct_type.a = converted @__global_init.%.loc5_28, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_16.2: %empty_tuple.type = struct_access %.loc5_16.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_16.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_16.3: %pattern_type.e85 = struct_pattern (%b.patt) [concrete = constants.%.140]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %struct_type.a.c.bac: type = struct_type {.a: %i32, .c: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.dea: type = pattern_type %struct_type.a.c.bac [concrete]
// CHECK:STDOUT:   %.9b8: %pattern_type.dea = struct_pattern (%b.patt, %d.patt) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %struct_type.a.c.bbc: type = struct_type {.a: Core.IntLiteral, .c: bool} [concrete]
// CHECK:STDOUT:   %struct.7e6: %struct_type.a.c.bbc = struct_value (%int_1.5b8, %true) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.7f4: %struct_type.a.c.bac = struct_value (%int_1.0c6, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_53.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_53.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_53.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_53.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_53.2: %i32 = converted @__global_init.%int_1, %.loc5_53.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %struct: %struct_type.a.c.bac = struct_value (%.loc5_53.2, @__global_init.%true) [concrete = constants.%struct.7f4]
// CHECK:STDOUT:   %.loc5_31.1: %struct_type.a.c.bac = converted @__global_init.%.loc5, %struct [concrete = constants.%struct.7f4]
// CHECK:STDOUT:   %.loc5_31.2: %i32 = struct_access %.loc5_31.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_31.3: bool = struct_access %.loc5_31.1, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc5_31.2
// CHECK:STDOUT:   %.loc5_27: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = wrapper_binding d, %.loc5_31.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %.loc5_31.4: %pattern_type.dea = struct_pattern (%b.patt, %d.patt) [concrete = constants.%.9b8]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc5: %struct_type.a.c.bbc = struct_literal (%int_1, %true) [concrete = constants.%struct.7e6]
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_23: %empty_tuple.type = converted @__global_init.%.loc5_22, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.a = struct_value (%.loc5_23) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_11.1: %struct_type.a = converted @__global_init.%.loc5_23, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_11.2: %empty_tuple.type = struct_access %.loc5_11.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_11.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %.loc5_11.3: %pattern_type.e85 = struct_pattern (%a.patt) [concrete = constants.%.e73]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %struct_type.a.b.56f: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b.56f [concrete]
// CHECK:STDOUT:   %.8fc: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f6e: type = struct_type {.a: Core.IntLiteral, .b: bool} [concrete]
// CHECK:STDOUT:   %struct.1e1: %struct_type.a.b.f6e = struct_value (%int_1.5b8, %true) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.f1c: %struct_type.a.b.56f = struct_value (%int_1.0c6, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_43.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_43.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_43.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_43.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_43.2: %i32 = converted @__global_init.%int_1, %.loc5_43.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %struct: %struct_type.a.b.56f = struct_value (%.loc5_43.2, @__global_init.%true) [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_21.1: %struct_type.a.b.56f = converted @__global_init.%.loc5, %struct [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_21.2: %i32 = struct_access %.loc5_21.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_21.3: bool = struct_access %.loc5_21.1, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc5_21.2
// CHECK:STDOUT:   %.loc5_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc5_21.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_21.4: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.8fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc5: %struct_type.a.b.f6e = struct_literal (%int_1, %true) [concrete = constants.%struct.1e1]
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
// CHECK:STDOUT:   %empty_tuple.loc5_34: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.1: %empty_tuple.type = converted @__global_init.%.loc5_34, %empty_tuple.loc5_34 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_43: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.2: %empty_tuple.type = converted @__global_init.%.loc5_43, %empty_tuple.loc5_43 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.a.c = struct_value (%.loc5_44.1, %.loc5_44.2) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_23.1: %struct_type.a.c = converted @__global_init.%.loc5_44, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_23.2: %empty_tuple.type = struct_access %.loc5_23.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_23.3: %empty_tuple.type = struct_access %.loc5_23.1, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_23.2
// CHECK:STDOUT:   %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_22.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_22.3: type = converted %.loc5_22.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %empty_tuple.type = wrapper_binding c, %.loc5_23.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %.loc5_23.4: %pattern_type.f10 = struct_pattern (%b.patt, %c.patt) [concrete = constants.%.fcd]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %f32.c73: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.240: type = pattern_type %f32.c73 [concrete]
// CHECK:STDOUT:   %m.patt: %pattern_type.240 = value_binding_pattern m [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.831 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %struct_type.i.j.da0: type = struct_type {.i: %f32.c73, .j: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.d8f: type = pattern_type %struct_type.i.j.da0 [concrete]
// CHECK:STDOUT:   %.1c4: %pattern_type.d8f = struct_pattern (%m.patt, %n.patt) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.68e: type = struct_type {.a: %i32, .b: %struct_type.i.j.da0} [concrete]
// CHECK:STDOUT:   %pattern_type.162: type = pattern_type %struct_type.a.b.68e [concrete]
// CHECK:STDOUT:   %.df9: %pattern_type.162 = struct_pattern (%x.patt, %.1c4) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %float.cce: Core.FloatLiteral = float_literal_value 20e-1 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %struct_type.i.j.e80: type = struct_type {.i: Core.FloatLiteral, .j: bool} [concrete]
// CHECK:STDOUT:   %struct.050: %struct_type.i.j.e80 = struct_value (%float.cce, %true) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.1a6: type = struct_type {.a: Core.IntLiteral, .b: %struct_type.i.j.e80} [concrete]
// CHECK:STDOUT:   %struct.293: %struct_type.a.b.1a6 = struct_value (%int_1.5b8, %struct.050) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.48f: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.48f) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet.48f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ed3: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.c73)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2fa: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fcd: %ImplicitAs.type.ed3 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.2fa) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.c73, %ImplicitAs.facet.fcd) [concrete]
// CHECK:STDOUT:   %.7fe: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7cb, %ImplicitAs.facet.fcd [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.cce, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c56: <bound method> = bound_method %float.cce, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.1e2: %f32.c73 = float_value 2 [concrete]
// CHECK:STDOUT:   %struct.5b4: %struct_type.i.j.da0 = struct_value (%float.1e2, %true) [concrete]
// CHECK:STDOUT:   %struct.2a7: %struct_type.a.b.68e = struct_value (%int_1.0c6, %struct.5b4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.063) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d4f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.c4c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0.loc5_90: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_90.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0.loc5_90 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5_90: <specific function> = specific_function %impl.elem0.loc5_90, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_90.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn.loc5_90 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_90.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_90.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_90.2: %i32 = converted @__global_init.%int_1, %.loc5_90.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc5_89: %.7fe = impl_witness_access constants.%ImplicitAs.impl_witness.2fa, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.b37]
// CHECK:STDOUT:   %bound_method.loc5_89.1: <bound method> = bound_method @__global_init.%float, %impl.elem0.loc5_89 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5_89: <specific function> = specific_function %impl.elem0.loc5_89, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_89.2: <bound method> = bound_method @__global_init.%float, %specific_fn.loc5_89 [concrete = constants.%bound_method.c56]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.c73 = call %bound_method.loc5_89.2(@__global_init.%float) [concrete = constants.%float.1e2]
// CHECK:STDOUT:   %.loc5_89.1: %f32.c73 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.1e2]
// CHECK:STDOUT:   %.loc5_89.2: %f32.c73 = converted @__global_init.%float, %.loc5_89.1 [concrete = constants.%float.1e2]
// CHECK:STDOUT:   %struct.loc5_89: %struct_type.i.j.da0 = struct_value (%.loc5_89.2, @__global_init.%true) [concrete = constants.%struct.5b4]
// CHECK:STDOUT:   %.loc5_90.3: %struct_type.i.j.da0 = converted @__global_init.%.loc5_89, %struct.loc5_89 [concrete = constants.%struct.5b4]
// CHECK:STDOUT:   %struct.loc5_90: %struct_type.a.b.68e = struct_value (%.loc5_90.2, %.loc5_90.3) [concrete = constants.%struct.2a7]
// CHECK:STDOUT:   %.loc5_51.1: %struct_type.a.b.68e = converted @__global_init.%.loc5_90, %struct.loc5_90 [concrete = constants.%struct.2a7]
// CHECK:STDOUT:   %.loc5_51.2: %i32 = struct_access %.loc5_51.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_51.3: %struct_type.i.j.da0 = struct_access %.loc5_51.1, element1 [concrete = constants.%struct.5b4]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc5_51.2
// CHECK:STDOUT:   %.loc5_51.4: %f32.c73 = struct_access %.loc5_51.3, element0 [concrete = constants.%float.1e2]
// CHECK:STDOUT:   %.loc5_51.5: bool = struct_access %.loc5_51.3, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.c73 [concrete = constants.%f32.c73]
// CHECK:STDOUT:   %m: %f32.c73 = wrapper_binding m, %.loc5_51.4
// CHECK:STDOUT:   %.loc5_46: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %n: bool = wrapper_binding n, %.loc5_51.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %m.patt: %pattern_type.240 = value_binding_pattern m [concrete = constants.%m.patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.831 = value_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     %.loc5_50: %pattern_type.d8f = struct_pattern (%m.patt, %n.patt) [concrete = constants.%.1c4]
// CHECK:STDOUT:     %.loc5_51.6: %pattern_type.162 = struct_pattern (%x.patt, %.loc5_50) [concrete = constants.%.df9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 20e-1 [concrete = constants.%float.cce]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc5_89: %struct_type.i.j.e80 = struct_literal (%float, %true) [concrete = constants.%struct.050]
// CHECK:STDOUT:   %.loc5_90: %struct_type.a.b.1a6 = struct_literal (%int_1, %.loc5_89) [concrete = constants.%struct.293]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.831 = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %.60e: %pattern_type.7b0 = struct_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_37.1: ref %struct_type.a.b = temporary @__global_init.%.loc7, @__global_init.%F.call
// CHECK:STDOUT:   %.loc7_37.2: ref %i32 = struct_access %.loc7_37.1, element0
// CHECK:STDOUT:   %.loc7_37.3: ref bool = struct_access %.loc7_37.1, element1
// CHECK:STDOUT:   %.loc7_37.4: %i32 = acquire_value %.loc7_37.2
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc7_37.4
// CHECK:STDOUT:   %.loc7_37.5: bool = acquire_value %.loc7_37.3
// CHECK:STDOUT:   %.loc7_27: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %y: bool = wrapper_binding y, %.loc7_37.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.831 = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc7_31: %pattern_type.7b0 = struct_pattern (%x.patt, %y.patt) [concrete = constants.%.60e]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.8fc: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_27.1: ref %struct_type.a.b = temporary @__global_init.%.loc7, @__global_init.%F.call
// CHECK:STDOUT:   %.loc7_27.2: ref %i32 = struct_access %.loc7_27.1, element0
// CHECK:STDOUT:   %.loc7_27.3: ref bool = struct_access %.loc7_27.1, element1
// CHECK:STDOUT:   %.loc7_27.4: %i32 = acquire_value %.loc7_27.2
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc7_27.4
// CHECK:STDOUT:   %.loc7_27.5: bool = acquire_value %.loc7_27.3
// CHECK:STDOUT:   %.loc7_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc7_27.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7_21: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.8fc]
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
// CHECK:STDOUT:   %struct_type.a.b.56f: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b.56f [concrete]
// CHECK:STDOUT:   %.0db: ref %i32 = struct_access file.%s.var, element0 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.91c: ref bool = struct_access file.%s.var, element1 [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.831 = value_binding_pattern y [concrete]
// CHECK:STDOUT:   %.60e: %pattern_type.7b0 = struct_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_35.1: ref %i32 = struct_access @__global_init.%s.ref, element0 [concrete = constants.%.0db]
// CHECK:STDOUT:   %.loc7_35.2: ref bool = struct_access @__global_init.%s.ref, element1 [concrete = constants.%.91c]
// CHECK:STDOUT:   %.loc7_35.3: %i32 = acquire_value %.loc7_35.1
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %.loc7_35.3
// CHECK:STDOUT:   %.loc7_35.4: bool = acquire_value %.loc7_35.2
// CHECK:STDOUT:   %.loc7_27: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %y: bool = wrapper_binding y, %.loc7_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.831 = value_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc7_31: %pattern_type.7b0 = struct_pattern (%x.patt, %y.patt) [concrete = constants.%.60e]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %s.ref: ref %struct_type.a.b.56f = name_ref s, file.%s [concrete = file.%s.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.56f: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b.56f [concrete]
// CHECK:STDOUT:   %.0db: ref %i32 = struct_access file.%s.var, element0 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.91c: ref bool = struct_access file.%s.var, element1 [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.8fc: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc7_25.1: ref %i32 = struct_access @__global_init.%s.ref, element0 [concrete = constants.%.0db]
// CHECK:STDOUT:   %.loc7_25.2: ref bool = struct_access @__global_init.%s.ref, element1 [concrete = constants.%.91c]
// CHECK:STDOUT:   %.loc7_25.3: %i32 = acquire_value %.loc7_25.1
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc7_25.3
// CHECK:STDOUT:   %.loc7_25.4: bool = acquire_value %.loc7_25.2
// CHECK:STDOUT:   %.loc7_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc7_25.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7_21: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.8fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %s.ref: ref %struct_type.a.b.56f = name_ref s, file.%s [concrete = file.%s.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unordered.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %struct_type.a.c: type = struct_type {.a: %i32, .c: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.dea: type = pattern_type %struct_type.a.c [concrete]
// CHECK:STDOUT:   %.9b8: %pattern_type.dea = struct_pattern (%b.patt, %d.patt) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c.a: type = struct_type {.c: bool, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.da6: %struct_type.c.a = struct_value (%true, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.7f4: %struct_type.a.c = struct_value (%int_1.0c6, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_53.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_53.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_53.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_53.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_53.2: %i32 = converted @__global_init.%int_1, %.loc5_53.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %struct: %struct_type.a.c = struct_value (%.loc5_53.2, @__global_init.%true) [concrete = constants.%struct.7f4]
// CHECK:STDOUT:   %.loc5_31.1: %struct_type.a.c = converted @__global_init.%.loc5, %struct [concrete = constants.%struct.7f4]
// CHECK:STDOUT:   %.loc5_31.2: %i32 = struct_access %.loc5_31.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_31.3: bool = struct_access %.loc5_31.1, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc5_31.2
// CHECK:STDOUT:   %.loc5_27: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = wrapper_binding d, %.loc5_31.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %.loc5_31.4: %pattern_type.dea = struct_pattern (%b.patt, %d.patt) [concrete = constants.%.9b8]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5: %struct_type.c.a = struct_literal (%true, %int_1) [concrete = constants.%struct.da6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unordered_shorthand.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %.8fc: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: bool, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a63: %struct_type.b.a = struct_value (%true, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.f1c: %struct_type.a.b = struct_value (%int_1.0c6, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_43.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_43.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_43.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_43.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_43.2: %i32 = converted @__global_init.%int_1, %.loc5_43.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%.loc5_43.2, @__global_init.%true) [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_21.1: %struct_type.a.b = converted @__global_init.%.loc5, %struct [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_21.2: %i32 = struct_access %.loc5_21.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_21.3: bool = struct_access %.loc5_21.1, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc5_21.2
// CHECK:STDOUT:   %.loc5_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %.loc5_21.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5_21.4: %pattern_type.7b0 = struct_pattern (%a.patt, %b.patt) [concrete = constants.%.8fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5: %struct_type.b.a = struct_literal (%true, %int_1) [concrete = constants.%struct.a63]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unordered_mixed_syntax.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.7b0: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %.b79: %pattern_type.7b0 = struct_pattern (%a.patt, %c.patt) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: bool, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a63: %struct_type.b.a = struct_value (%true, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.f1c: %struct_type.a.b = struct_value (%int_1.0c6, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_48.1: <bound method> = bound_method @__global_init.%int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_48.2: <bound method> = bound_method @__global_init.%int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_48.2(@__global_init.%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_48.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_48.2: %i32 = converted @__global_init.%int_1, %.loc5_48.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%.loc5_48.2, @__global_init.%true) [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_26.1: %struct_type.a.b = converted @__global_init.%.loc5, %struct [concrete = constants.%struct.f1c]
// CHECK:STDOUT:   %.loc5_26.2: %i32 = struct_access %.loc5_26.1, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_26.3: bool = struct_access %.loc5_26.1, element1 [concrete = constants.%true]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc5_26.2
// CHECK:STDOUT:   %.loc5_22: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %c: bool = wrapper_binding c, %.loc5_26.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %.loc5_26.4: %pattern_type.7b0 = struct_pattern (%a.patt, %c.patt) [concrete = constants.%.b79]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5: %struct_type.b.a = struct_literal (%true, %int_1) [concrete = constants.%struct.a63]
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
// CHECK:STDOUT: --- fail_no_matching_field_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.810: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %.9cb: %pattern_type.810 = struct_pattern (%x.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc9: %pattern_type.810 = struct_pattern (%x.patt) [concrete = constants.%.9cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %s.ref: ref %struct_type.a.b = name_ref s, file.%s [concrete = file.%s.var]
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
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: bool} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%empty_tuple, %true) [concrete]
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
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_39: %struct_type.a.b = struct_literal (%.loc9_27, %true) [concrete = constants.%struct]
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
// CHECK:STDOUT: --- fail_incorrect_field_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %struct_type.a.715: type = struct_type {.a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %struct_type.a.715 [concrete]
// CHECK:STDOUT:   %.298: %pattern_type.e85 = struct_pattern (%x.patt) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.225 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc4_28: %empty_tuple.type = converted @__global_init.%.loc4_27, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc4_16: %pattern_type.e85 = struct_pattern (%x.patt) [concrete = constants.%.298]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_27: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc4_28: %struct_type.a.225 = struct_literal (%.loc4_27) [concrete = constants.%struct]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incorrect_scrutinee_type.carbon
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
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc11_16.1: %struct_type.a = converted @__global_init.%.loc11_27, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %.loc11_16.2: %pattern_type.e85 = struct_pattern (%x.patt) [concrete = constants.%.298]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_27: %tuple.type = tuple_literal (%.loc11_22, %.loc11_26) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

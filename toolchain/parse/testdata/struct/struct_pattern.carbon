// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/struct/struct_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/struct/struct_pattern.carbon

// --- struct_pattern_empty.carbon

let {} = {};

// --- struct_pattern_one_entry.carbon

let {.x = y: i32} = {.x = 1};

// --- struct_pattern_trailing_comma.carbon

let {.x = y: i32,} = {.x = 1};

// --- struct_pattern_two_entries.carbon

let {.x = y: i32, .m = n: i32} = {.x = 1, .m = 2};

// --- struct_value_literal_as_pattern.carbon
// TODO: Allow ExprPatterns inside a StructPattern
// CHECK:STDERR: struct_value_literal_as_pattern.carbon:[[@LINE+8]]:12: error: Expression pattern nested within a struct pattern [ExprPatternInStructPattern]
// CHECK:STDERR: let {.x = 1, .y = 2};
// CHECK:STDERR:            ^
// CHECK:STDERR:
// CHECK:STDERR: struct_value_literal_as_pattern.carbon:[[@LINE+4]]:20: error: Expression pattern nested within a struct pattern [ExprPatternInStructPattern]
// CHECK:STDERR: let {.x = 1, .y = 2};
// CHECK:STDERR:                    ^
// CHECK:STDERR:
let {.x = 1, .y = 2};

// --- struct_pattern_discard.carbon

let {.x = y: i32, _} = {.x = 1, .m = 2};

// --- struct_pattern_unnamed.carbon
// TODO: should be allowed in the presence of a designator
// CHECK:STDERR: struct_pattern_unnamed.carbon:[[@LINE+4]]:12: error: Unnamed binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [UnnamedBindingInStructPattern]
// CHECK:STDERR: let {.x = _: i32} = {.x = 1};
// CHECK:STDERR:            ^
// CHECK:STDERR:
let {.x = _: i32} = {.x = 1};

// --- struct_pattern_shorthand.carbon

let {x: i32, y: i32} = {.x = 1, .y = 2};

// --- struct_pattern_shorthand_ref.carbon

let {ref x: i32, y: i32} = {.x = 1, .y = 2};

// --- struct_pattern_shorthand_mixed.carbon

let {.x = y: i32, m: i32} = {.x = 1, .m = 2};

// --- fail_struct_pattern_comma_only.carbon

// CHECK:STDERR: fail_struct_pattern_comma_only.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {,} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {,} = {};

// --- fail_struct_pattern_missing_designator.carbon

// CHECK:STDERR: fail_struct_pattern_missing_designator.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {= y: i32} = {.x = 1};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {= y: i32} = {.x = 1};

// --- fail_struct_pattern_invalid_designator.carbon

// CHECK:STDERR: fail_struct_pattern_invalid_designator.carbon:[[@LINE+4]]:7: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: let {.7 = y: i32} = {.x = 1};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.7 = y: i32} = {.x = 1};

// --- fail_struct_pattern_designator_only.carbon

// CHECK:STDERR: fail_struct_pattern_designator_only.carbon:[[@LINE+4]]:8: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x} = {.x = 1};
// CHECK:STDERR:        ^
// CHECK:STDERR:
let {.x} = {.x = 1};

// --- fail_struct_type_literal_as_pattern.carbon

// CHECK:STDERR: fail_struct_type_literal_as_pattern.carbon:[[@LINE+4]]:8: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x: i32} = {.x = 1};
// CHECK:STDERR:        ^
// CHECK:STDERR:
let {.x: i32} = {.x = 1};

// --- fail_struct_pattern_period_only.carbon

// CHECK:STDERR: fail_struct_pattern_period_only.carbon:[[@LINE+4]]:7: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: let {.} = {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.} = {};

// --- fail_struct_pattern_repeated_comma.carbon

// CHECK:STDERR: fail_struct_pattern_repeated_comma.carbon:[[@LINE+4]]:18: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {.x = x: i32,,} = {.x = 1};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {.x = x: i32,,} = {.x = 1};

// --- fail_struct_pattern_missing_comma.carbon

// CHECK:STDERR: fail_struct_pattern_missing_comma.carbon:[[@LINE+4]]:21: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {.x = x: i32 .y = y: i32} = {.x = 1, .y = 2};
// CHECK:STDERR:                     ^
// CHECK:STDERR:
let {.x = x: i32 .y = y: i32} = {.x = 1, .y = 2};

// --- fail_struct_pattern_missing_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_missing_pattern.carbon:[[@LINE+4]]:11: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {.x = } = {.x = 1};
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {.x = } = {.x = 1};

// --- fail_struct_pattern_shorthand_missing_identifier.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_identifier.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {:i32} = {.x = 1};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {:i32} = {.x = 1};

// --- fail_struct_pattern_shorthand_missing_expression.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_expression.carbon:[[@LINE+4]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: let {x: } = {.x = 1};
// CHECK:STDERR:         ^
// CHECK:STDERR:
let {x: } = {.x = 1};

// --- fail_struct_pattern_shorthand_missing_comma.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_comma.carbon:[[@LINE+4]]:13: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {x: i32 y: i32} = {.x=1, .y=1};
// CHECK:STDERR:             ^
// CHECK:STDERR:
let {x: i32 y: i32} = {.x=1, .y=1};

// --- fail_struct_pattern_missing_equals.carbon

// CHECK:STDERR: fail_struct_pattern_missing_equals.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x x:i32} = {.x=0};
// CHECK:STDERR:         ^
// CHECK:STDERR:
let {.x x:i32} = {.x=0};

// --- fail_struct_pattern_token_after_underscore.carbon

// CHECK:STDERR: fail_struct_pattern_token_after_underscore.carbon:[[@LINE+4]]:20: error: unexpected token `,` after `_` in struct pattern, expected `}` [ExpectedCloseAfterUnderscore]
// CHECK:STDERR: let {.x = x: i32, _,} = {.x=1, .y = 2};
// CHECK:STDERR:                    ^
// CHECK:STDERR:
let {.x = x: i32, _,} = {.x=1, .y = 2};

// --- fail_struct_pattern_extra_token_end.carbon

// CHECK:STDERR: fail_struct_pattern_extra_token_end.carbon:[[@LINE+4]]:18: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {.x = x: i32 foo, .y = y: i32} = {.x = 1, .y = 2};
// CHECK:STDERR:                  ^~~
// CHECK:STDERR:
let {.x = x: i32 foo, .y = y: i32} = {.x = 1, .y = 2};

// --- fail_struct_pattern_extra_token_start.carbon

// CHECK:STDERR: fail_struct_pattern_extra_token_start.carbon:[[@LINE+4]]:26: error: Expression pattern nested within a struct pattern [ExprPatternInStructPattern]
// CHECK:STDERR: let {.x = x: i32, foo .y = y: i32} = {.x = 1, .y = 2};
// CHECK:STDERR:                          ^
// CHECK:STDERR:
let {.x = x: i32, foo .y = y: i32} = {.x = 1, .y = 2};

// --- fail_struct_pattern_shorthand_extra_token_end.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_extra_token_end.carbon:[[@LINE+4]]:13: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {x: i32 foo, y: i32} = {.x = 1, .y = 2};
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
let {x: i32 foo, y: i32} = {.x = 1, .y = 2};

// --- file_struct_pattern_shorthand_extra_token_start.carbon

// CHECK:STDERR: file_struct_pattern_shorthand_extra_token_start.carbon:[[@LINE+4]]:18: error: Expression pattern nested within a struct pattern [ExprPatternInStructPattern]
// CHECK:STDERR: let {x: i32, foo y: i32} = {.x = 1, .y = 2};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {x: i32, foo y: i32} = {.x = 1, .y = 2};

// --- file_struct_pattern_invalid_introducer.carbon

// CHECK:STDERR: file_struct_pattern_invalid_introducer.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {!x = x: i32} = {.x = 1};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {!x = x: i32} = {.x = 1};

// --- fail_struct_pattern_extra_token_middle.carbon

// recover on `=`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo = x: ()} = {.x = ()};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo = x: ()} = {.x = ()};

// no equals or comma - recover on `}`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: ()} = {.x = ()};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: ()} = {.x = ()};

// no equals, trailing comma - recover on `,`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: (),} = {.x = ()};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: (),} = {.x = ()};

// no equals, second field - recover on `,`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: (), .y = y: ()} = {.x = (), .y = ()};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: (), .y = y: ()} = {.x = (), .y = ()};

// --- fail_struct_pattern_invalid_field.carbon

// CHECK:STDERR: fail_struct_pattern_invalid_field.carbon:[[@LINE+4]]:12: error: Expression pattern nested within a struct pattern [ExprPatternInStructPattern]
// CHECK:STDERR: let {"foo" = , .x = x:()} = {.foo = 1, .x = ()};
// CHECK:STDERR:            ^
// CHECK:STDERR:
let {"foo" = , .x = x:()} = {.foo = 1, .x = ()};

// --- fail_struct_pattern_nested_tuple_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_tuple_pattern.carbon:[[@LINE+4]]:14: error: Tuple pattern nested within a struct pattern [NestedPatternListInStructPattern]
// CHECK:STDERR: var {x: i32, (y: i32, z: i32)} = ();
// CHECK:STDERR:              ^
// CHECK:STDERR:
var {x: i32, (y: i32, z: i32)} = ();

// --- fail_struct_pattern_nested_var_tuple_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_var_tuple_pattern.carbon:[[@LINE+4]]:18: error: Tuple pattern nested within a struct pattern [NestedPatternListInStructPattern]
// CHECK:STDERR: let {x: i32, var (y: i32, z: i32)} = ();
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {x: i32, var (y: i32, z: i32)} = ();

// --- fail_struct_pattern_nested_struct_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_struct_pattern.carbon:[[@LINE+4]]:13: error: Struct pattern nested within a struct pattern [NestedPatternListInStructPattern]
// CHECK:STDERR: let{x: i32, {y: i32, z: i32}} = ();
// CHECK:STDERR:             ^
// CHECK:STDERR:
let{x: i32, {y: i32, z: i32}} = ();

// --- fail_struct_pattern_unused_shorthand.carbon

// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:7: error: Unnamed binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [UnnamedBindingInStructPattern]
// CHECK:STDERR: var {_: i32} = ();
// CHECK:STDERR:       ^
// CHECK:STDERR:
var {_: i32} = ();
// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:11: error: Unnamed binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [UnnamedBindingInStructPattern]
// CHECK:STDERR: let {ref _: i32} = ();
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {ref _: i32} = ();
// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:11: error: Unnamed binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [UnnamedBindingInStructPattern]
// CHECK:STDERR: let {var _: i32} = ();
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {var _: i32} = ();

// CHECK:STDOUT: - filename: struct_pattern_empty.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_one_entry.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_trailing_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_two_entries.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'm'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'm'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 29},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_value_literal_as_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_discard.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:         {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'm'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_unnamed.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_shorthand.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_shorthand_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: struct_pattern_shorthand_mixed.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'm'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'm'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_comma_only.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ',', has_error: yes},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_invalid_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: '7', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_designator_only.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_type_literal_as_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_period_only.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: '}', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_repeated_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ',', has_error: yes},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_expression.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_equals.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'x', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_token_after_underscore.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:         {kind: 'UnderscoreName', text: '_', has_error: yes},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_end.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 15},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 29},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_start.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 12},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_extra_token_end.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: file_struct_pattern_shorthand_extra_token_start.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: file_struct_pattern_invalid_introducer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '!', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_middle.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'foo', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'foo', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'foo', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '.', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 30},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_invalid_field.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"foo"'},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructPatternDesignatedField', text: '=', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'foo'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructLiteralField', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_tuple_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'z'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 15},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_var_tuple_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'z'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 16},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_struct_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:           {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'z'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPattern', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 15},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_struct_pattern_unused_shorthand.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'StructPatternStart', text: '{'},
// CHECK:STDOUT:             {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructPattern', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]

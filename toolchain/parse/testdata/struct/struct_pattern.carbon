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

let {.x = y: i32} = {};

// --- struct_pattern_trailing_comma.carbon

let {.x = y: i32,} = {};

// --- struct_pattern_two_entries.carbon

let {.x = y: i32, .m = n: i32} = {};

// --- struct_value_literal_as_pattern.carbon

let {.x = 1, .y = 2};

// --- struct_pattern_discard.carbon

let {.x = y: i32, _} = {};

// --- struct_pattern_unnamed.carbon

let {.x = _: i32} = {};

// --- struct_pattern_shorthand.carbon

let {x: i32, y: i32} = {};

// --- struct_pattern_shorthand_ref.carbon

let {ref x: i32, y: i32} = {};

// --- struct_pattern_shorthand_mixed.carbon

let {.x = y: i32, m: i32} = {};

// --- struct_pattern_designated_tuple.carbon

let{x: i32, .y = (a: i32, b: i32)} = {};

// --- struct_pattern_designated_struct.carbon

let{x: i32, .y = {a: i32, b: i32}} = {};

// --- fail_struct_pattern_comma_only.carbon

// CHECK:STDERR: fail_struct_pattern_comma_only.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {,} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {,} = {};

// --- fail_struct_pattern_missing_designator.carbon

// CHECK:STDERR: fail_struct_pattern_missing_designator.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {= y: i32} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {= y: i32} = {};

// --- fail_struct_pattern_invalid_designator.carbon

// CHECK:STDERR: fail_struct_pattern_invalid_designator.carbon:[[@LINE+4]]:7: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: let {.7 = y: i32} = {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.7 = y: i32} = {};

// --- fail_struct_pattern_designator_only.carbon

// CHECK:STDERR: fail_struct_pattern_designator_only.carbon:[[@LINE+4]]:8: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x} = {};
// CHECK:STDERR:        ^
// CHECK:STDERR:
let {.x} = {};

// --- fail_struct_type_literal_as_pattern.carbon

// CHECK:STDERR: fail_struct_type_literal_as_pattern.carbon:[[@LINE+4]]:8: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x: i32} = {};
// CHECK:STDERR:        ^
// CHECK:STDERR:
let {.x: i32} = {};

// --- fail_struct_pattern_period_only.carbon

// CHECK:STDERR: fail_struct_pattern_period_only.carbon:[[@LINE+4]]:7: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR: let {.} = {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
let {.} = {};

// --- fail_struct_pattern_repeated_comma.carbon

// CHECK:STDERR: fail_struct_pattern_repeated_comma.carbon:[[@LINE+4]]:18: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {.x = x: i32,,} = {};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {.x = x: i32,,} = {};

// --- fail_struct_pattern_missing_comma.carbon

// CHECK:STDERR: fail_struct_pattern_missing_comma.carbon:[[@LINE+4]]:21: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {.x = x: i32 .y = y: i32} = {};
// CHECK:STDERR:                     ^
// CHECK:STDERR:
let {.x = x: i32 .y = y: i32} = {};

// --- fail_struct_pattern_missing_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_missing_pattern.carbon:[[@LINE+4]]:11: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {.x = } = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {.x = } = {};

// --- fail_struct_pattern_shorthand_missing_identifier.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_identifier.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {:i32} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {:i32} = {};

// --- fail_struct_pattern_shorthand_missing_expression.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_expression.carbon:[[@LINE+4]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: let {x: } = {};
// CHECK:STDERR:         ^
// CHECK:STDERR:
let {x: } = {};

// --- fail_struct_pattern_shorthand_missing_comma.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_missing_comma.carbon:[[@LINE+4]]:13: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {x: i32 y: i32} = {};
// CHECK:STDERR:             ^
// CHECK:STDERR:
let {x: i32 y: i32} = {};

// --- fail_struct_pattern_missing_equals.carbon

// CHECK:STDERR: fail_struct_pattern_missing_equals.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x x:i32} = {};
// CHECK:STDERR:         ^
// CHECK:STDERR:
let {.x x:i32} = {};

// --- fail_struct_pattern_token_after_underscore.carbon

// CHECK:STDERR: fail_struct_pattern_token_after_underscore.carbon:[[@LINE+4]]:20: error: unexpected token `,` after `_` in struct pattern, expected `}` [ExpectedCloseAfterUnderscore]
// CHECK:STDERR: let {.x = x: i32, _,} = {};
// CHECK:STDERR:                    ^
// CHECK:STDERR:
let {.x = x: i32, _,} = {};

// --- fail_struct_pattern_underscore_not_last.carbon

// CHECK:STDERR: fail_struct_pattern_underscore_not_last.carbon:[[@LINE+4]]:6: error: unexpected token `,` after `_` in struct pattern, expected `}` [ExpectedCloseAfterUnderscore]
// CHECK:STDERR: let{_, .x = x: i32} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let{_, .x = x: i32} = {};

// --- fail_struct_pattern_extra_token_end.carbon

// CHECK:STDERR: fail_struct_pattern_extra_token_end.carbon:[[@LINE+4]]:18: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {.x = x: i32 foo, .y = y: i32} = {};
// CHECK:STDERR:                  ^~~
// CHECK:STDERR:
let {.x = x: i32 foo, .y = y: i32} = {};

// --- fail_struct_pattern_extra_token_start.carbon

// CHECK:STDERR: fail_struct_pattern_extra_token_start.carbon:[[@LINE+4]]:26: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {.x = x: i32, foo .y = y: i32} = {};
// CHECK:STDERR:                          ^
// CHECK:STDERR:
let {.x = x: i32, foo .y = y: i32} = {};

// --- fail_struct_pattern_shorthand_extra_token_end.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_extra_token_end.carbon:[[@LINE+4]]:13: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {x: i32 foo, y: i32} = {};
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
let {x: i32 foo, y: i32} = {};

// --- fail_struct_pattern_shorthand_extra_token_start.carbon

// CHECK:STDERR: fail_struct_pattern_shorthand_extra_token_start.carbon:[[@LINE+4]]:18: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {x: i32, foo y: i32} = {};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {x: i32, foo y: i32} = {};

// --- fail_struct_pattern_invalid_introducer.carbon

// CHECK:STDERR: fail_struct_pattern_invalid_introducer.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
// CHECK:STDERR: let {!x = x: i32} = {};
// CHECK:STDERR:      ^
// CHECK:STDERR:
let {!x = x: i32} = {};

// --- fail_struct_pattern_extra_token_middle.carbon

// recover on `=`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo = x: ()} = {};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo = x: ()} = {};

// no equals or comma - recover on `}`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: ()} = {};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: ()} = {};

// no equals, trailing comma - recover on `,`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: (),} = {};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: (),} = {};

// no equals, second field - recover on `,`
// CHECK:STDERR: fail_struct_pattern_extra_token_middle.carbon:[[@LINE+4]]:9: error: expected `= value` after `.field` [ExpectedStructPatternDesignatedField]
// CHECK:STDERR: let {.x foo x: (), .y = y: ()} = {};
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
let {.x foo x: (), .y = y: ()} = {};

// --- fail_struct_pattern_invalid_field.carbon

// CHECK:STDERR: fail_struct_pattern_invalid_field.carbon:[[@LINE+4]]:12: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: let {"foo" = , .x = x:()} = {};
// CHECK:STDERR:            ^
// CHECK:STDERR:
let {"foo" = , .x = x:()} = {};

// --- fail_struct_pattern_nested_tuple_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_tuple_pattern.carbon:[[@LINE+4]]:14: error: Tuple pattern in shorthand struct pattern field. Use `.field = (...)` instead [NestedTuplePatternInStructPatternShortField]
// CHECK:STDERR: var {x: i32, (y: i32, z: i32)} = {};
// CHECK:STDERR:              ^
// CHECK:STDERR:
var {x: i32, (y: i32, z: i32)} = {};

// --- fail_struct_pattern_nested_var_tuple_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_var_tuple_pattern.carbon:[[@LINE+4]]:18: error: Tuple pattern in shorthand struct pattern field. Use `.field = (...)` instead [NestedTuplePatternInStructPatternShortField]
// CHECK:STDERR: let {x: i32, var (y: i32, z: i32)} = {};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
let {x: i32, var (y: i32, z: i32)} = {};

// --- fail_struct_pattern_nested_struct_pattern.carbon

// CHECK:STDERR: fail_struct_pattern_nested_struct_pattern.carbon:[[@LINE+4]]:13: error: Struct pattern in shorthand struct pattern field. Use `.field = {...}` instead [NestedStructPatternStructPatternShortField]
// CHECK:STDERR: let{x: i32, {y: i32, z: i32}} = {};
// CHECK:STDERR:             ^
// CHECK:STDERR:
let{x: i32, {y: i32, z: i32}} = {};

// --- fail_struct_pattern_unused_shorthand.carbon

// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:7: error: Anonymous binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [AnonymousBindingInStructPattern]
// CHECK:STDERR: var {_: i32} = {};
// CHECK:STDERR:       ^
// CHECK:STDERR:
var {_: i32} = {};
// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:11: error: Anonymous binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [AnonymousBindingInStructPattern]
// CHECK:STDERR: let {ref _: i32} = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {ref _: i32} = {};
// CHECK:STDERR: fail_struct_pattern_unused_shorthand.carbon:[[@LINE+4]]:11: error: Anonymous binding found in struct pattern. Use `.field = _: field_type` or `unused field: field_type` [AnonymousBindingInStructPattern]
// CHECK:STDERR: let {var _: i32} = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
let {var _: i32} = {};

// CHECK:STDOUT: - filename: struct_pattern_empty.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_one_entry.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_trailing_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_two_entries.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'm'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_value_literal_as_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_discard.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ ├─UnderscoreName '_'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_unnamed.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_shorthand_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_shorthand_mixed.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'm'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_designated_tuple.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─TuplePattern ')'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_pattern_designated_struct.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─StructPattern '}'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_comma_only.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─InvalidParse '=' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_invalid_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature '7' has_error
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '=' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_designator_only.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_type_literal_as_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse ':' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_period_only.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature '}' has_error
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_repeated_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ ├─InvalidParse ',' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ │ ├─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '=' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─InvalidParse ':' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_expression.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_missing_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_missing_equals.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'x' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_token_after_underscore.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ ├─UnderscoreName '_' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_underscore_not_last.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─UnderscoreName '_' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_end.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_start.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_extra_token_end.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_shorthand_extra_token_start.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_invalid_introducer.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─InvalidParse '!' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_extra_token_middle.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '=' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'foo' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'foo' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ ├─InvalidParse 'foo' has_error
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '.' has_error
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_invalid_field.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ ├─StringLiteral '"foo"'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_tuple_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─TuplePattern ')'
// CHECK:STDOUT:   │ │ ╭─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_var_tuple_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ╭─TuplePattern ')'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_nested_struct_pattern.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_struct_pattern_unused_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │   │ ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ╭─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │ │ ╭─RefBindingName 'ref' has_error
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─StructPattern '}' has_error
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/udiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/udiv.carbon

// --- int_div.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.udiv";

var arr: array(i32, Div(3, 2));
let arr_p: array(i32, 1)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return Div(a, b);
  //@dump-sem-ir-end
}

// --- overflow.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.udiv";
fn Sub(a: i32, b: i32) -> i32 = "int.usub";
fn Negate(a: i32) -> i32 = "int.unegate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 is OK.
let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

library "[[@TEST_NAME]]";

fn Div(a: i32, b: i32) -> i32 = "int.udiv";

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Div(1, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: i32 = Div(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Div(0, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let b: i32 = Div(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [concrete]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: %Div.type = name_ref Div, file.%Div.decl [concrete = constants.%Div]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %Div.call: init %i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %Div.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT: --- Associated NodeIds
// CHECK:STDOUT: InstKind AddrOf: PrefixOperatorAmpId
// CHECK:STDOUT: InstKind ArrayType: ArrayExprId
// CHECK:STDOUT: InstKind Assign: VariablePatternId
// CHECK:STDOUT: InstKind BoundMethod: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind Call: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind Converted: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind FunctionDecl: BuiltinFunctionDefinitionStartId, FunctionDefinitionStartId
// CHECK:STDOUT: InstKind ImplWitnessAccess: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind InitForm: IntTypeLiteralId
// CHECK:STDOUT: InstKind IntValue: IntLiteralId
// CHECK:STDOUT: InstKind NameBindingDecl: LetBindingPatternId, VariablePatternId
// CHECK:STDOUT: InstKind NameRef: IdentifierNameExprId
// CHECK:STDOUT: InstKind OutParam: IntTypeLiteralId
// CHECK:STDOUT: InstKind OutParamPattern: IntTypeLiteralId
// CHECK:STDOUT: InstKind PointerType: PostfixOperatorStarId
// CHECK:STDOUT: InstKind RefBinding: VarBindingPatternId
// CHECK:STDOUT: InstKind RefBindingPattern: VarBindingPatternId
// CHECK:STDOUT: InstKind ReturnExpr: ReturnStatementId
// CHECK:STDOUT: InstKind ReturnSlot: IntTypeLiteralId
// CHECK:STDOUT: InstKind ReturnSlotPattern: IntTypeLiteralId
// CHECK:STDOUT: InstKind SpecificFunction: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind SpliceBlock: ArrayExprId, PostfixOperatorStarId, VariablePatternId
// CHECK:STDOUT: InstKind TypeLiteral: IntTypeLiteralId
// CHECK:STDOUT: InstKind ValueBinding: LetBindingPatternId
// CHECK:STDOUT: InstKind ValueBindingPattern: LetBindingPatternId
// CHECK:STDOUT: InstKind ValueOfInitializer: CallExprId, IntLiteralId
// CHECK:STDOUT: InstKind ValueParam: LetBindingPatternId
// CHECK:STDOUT: InstKind ValueParamPattern: LetBindingPatternId
// CHECK:STDOUT: InstKind VarPattern: VariablePatternId
// CHECK:STDOUT: InstKind VarStorage: VariablePatternId
// CHECK:STDOUT: InstKind WrapperBindingPattern: LetBindingPatternId
// CHECK:STDOUT:

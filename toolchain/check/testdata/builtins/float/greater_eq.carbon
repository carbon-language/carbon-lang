// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/greater_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/greater_eq.carbon

// --- float_greater_eq.carbon

fn GreaterEq(a: f64, b: f64) -> bool = "float.greater_eq";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if GreaterEq(1.0, 2.0) then True else False);
  true_ as (if GreaterEq(1.0, 1.0) then True else False);
  true_ as (if GreaterEq(1.0, 0.0) then True else False);
  false_ as (if GreaterEq(Negate(1.0), 0.0) then True else False);
  true_ as (if GreaterEq(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return GreaterEq(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_greater_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %GreaterEq.type: type = fn_type @GreaterEq [concrete]
// CHECK:STDOUT:   %GreaterEq: %GreaterEq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %f64.d77, %b.param: %f64.d77) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GreaterEq.ref: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [concrete = constants.%GreaterEq]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %f64.d77 = name_ref b, %b
// CHECK:STDOUT:   %GreaterEq.call: init bool = call %GreaterEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %GreaterEq.call
// CHECK:STDOUT: }
// CHECK:STDOUT: --- Associated NodeIds
// CHECK:STDOUT: InstKind BlockArg: IfExprIfId
// CHECK:STDOUT: InstKind BoundMethod: RealLiteralId
// CHECK:STDOUT: InstKind Branch: IfExprIfId
// CHECK:STDOUT: InstKind BranchIf: IfExprIfId
// CHECK:STDOUT: InstKind BranchWithArg: IfExprIfId
// CHECK:STDOUT: InstKind Call: CallExprId, RealLiteralId
// CHECK:STDOUT: InstKind ClassDecl: ClassDefinitionStartId
// CHECK:STDOUT: InstKind CompleteTypeWitness: ClassDefinitionId
// CHECK:STDOUT: InstKind Converted: CallExprId, IfExprIfId, RealLiteralId
// CHECK:STDOUT: InstKind FloatLiteralValue: RealLiteralId
// CHECK:STDOUT: InstKind FunctionDecl: BuiltinFunctionDefinitionStartId, FunctionDefinitionStartId
// CHECK:STDOUT: InstKind ImplWitnessAccess: RealLiteralId
// CHECK:STDOUT: InstKind InitForm: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind NameRef: IdentifierNameExprId
// CHECK:STDOUT: InstKind OutParam: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind OutParamPattern: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind Return: FunctionDefinitionId
// CHECK:STDOUT: InstKind ReturnExpr: ReturnStatementId
// CHECK:STDOUT: InstKind ReturnSlot: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind ReturnSlotPattern: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind SpecificFunction: RealLiteralId
// CHECK:STDOUT: InstKind TypeLiteral: BoolTypeLiteralId, FloatTypeLiteralId
// CHECK:STDOUT: InstKind ValueBinding: LetBindingPatternId
// CHECK:STDOUT: InstKind ValueOfInitializer: RealLiteralId
// CHECK:STDOUT: InstKind ValueParam: LetBindingPatternId
// CHECK:STDOUT: InstKind ValueParamPattern: LetBindingPatternId
// CHECK:STDOUT: InstKind WrapperBindingPattern: LetBindingPatternId
// CHECK:STDOUT:

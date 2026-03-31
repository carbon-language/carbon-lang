// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/less.carbon

// --- float_less.carbon

fn Less(a: f64, b: f64) -> bool = "float.less";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1.0, 2.0) then True else False);
  false_ as (if Less(1.0, 1.0) then True else False);
  false_ as (if Less(1.0, 0.0) then True else False);
  true_ as (if Less(Negate(1.0), 0.0) then True else False);
  false_ as (if Less(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  //@dump-sem-ir-begin
  return Less(a, b);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- float_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %f64.d77, %b.param: %f64.d77) -> out %return.param: bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [concrete = constants.%Less]
// CHECK:STDOUT:   %a.ref: %f64.d77 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %f64.d77 = name_ref b, %b
// CHECK:STDOUT:   %Less.call: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %Less.call
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

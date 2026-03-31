// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/local.carbon

// --- local.carbon

library "[[@TEST_NAME]]";

fn F() -> () {
  var a: () = ();
  //@dump-sem-ir-begin
  alias b = a;
  //@dump-sem-ir-end
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %b: ref %empty_tuple.type = alias_binding b, %a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT: --- Associated NodeIds
// CHECK:STDOUT: InstKind AliasBinding: IdentifierNameNotBeforeSignatureId
// CHECK:STDOUT: InstKind Assign: VariablePatternId
// CHECK:STDOUT: InstKind Converted: ReturnStatementId, TupleLiteralId, VariablePatternId
// CHECK:STDOUT: InstKind FunctionDecl: FunctionDefinitionStartId
// CHECK:STDOUT: InstKind InitForm: TupleLiteralId
// CHECK:STDOUT: InstKind NameBindingDecl: VariablePatternId
// CHECK:STDOUT: InstKind NameRef: IdentifierNameExprId
// CHECK:STDOUT: InstKind OutParam: TupleLiteralId
// CHECK:STDOUT: InstKind OutParamPattern: TupleLiteralId
// CHECK:STDOUT: InstKind RefBinding: VarBindingPatternId
// CHECK:STDOUT: InstKind RefBindingPattern: VarBindingPatternId
// CHECK:STDOUT: InstKind ReturnExpr: ReturnStatementId
// CHECK:STDOUT: InstKind ReturnSlot: TupleLiteralId
// CHECK:STDOUT: InstKind ReturnSlotPattern: TupleLiteralId
// CHECK:STDOUT: InstKind SpliceBlock: TupleLiteralId
// CHECK:STDOUT: InstKind TupleInit: IdentifierNameExprId, TupleLiteralId
// CHECK:STDOUT: InstKind TupleLiteral: TupleLiteralId
// CHECK:STDOUT: InstKind VarPattern: VariablePatternId
// CHECK:STDOUT: InstKind VarStorage: VariablePatternId
// CHECK:STDOUT:

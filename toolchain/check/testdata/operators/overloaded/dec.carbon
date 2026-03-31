// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon

// This file was generated from unary_stmt.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Dec {
  fn Op[ref self: C]();
}

fn TestOp() {
  var c: C = {};
  //@dump-sem-ir-begin
  --c;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- dec.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Dec.impl_witness: <witness> = impl_witness @C.as.Dec.impl.%Dec.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op.type: type = fn_type @C.as.Dec.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Dec.impl.Op: %C.as.Dec.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %C, (%Dec.impl_witness) [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.874: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Dec.WithSelf.Op.type.874, %Dec.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %impl.elem0: %.4d8 = impl_witness_access constants.%Dec.impl_witness, element0 [concrete = constants.%C.as.Dec.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.Dec.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT: --- Associated NodeIds
// CHECK:STDOUT: InstKind Assign: VariablePatternId
// CHECK:STDOUT: InstKind BoundMethod: PrefixOperatorMinusMinusId
// CHECK:STDOUT: InstKind Call: PrefixOperatorMinusMinusId
// CHECK:STDOUT: InstKind ClassDecl: ClassDefinitionStartId
// CHECK:STDOUT: InstKind ClassInit: StructLiteralId
// CHECK:STDOUT: InstKind CompleteTypeWitness: ClassDefinitionId
// CHECK:STDOUT: InstKind Converted: VariablePatternId
// CHECK:STDOUT: InstKind FunctionDecl: FunctionDeclId, FunctionDefinitionStartId
// CHECK:STDOUT: InstKind ImplDecl: ImplDefinitionStartId
// CHECK:STDOUT: InstKind ImplWitness: ImplDefinitionStartId
// CHECK:STDOUT: InstKind ImplWitnessAccess: PrefixOperatorMinusMinusId
// CHECK:STDOUT: InstKind ImplWitnessTable: ImplDefinitionStartId
// CHECK:STDOUT: InstKind NameBindingDecl: VariablePatternId
// CHECK:STDOUT: InstKind NameRef: CoreNameExprId, IdentifierNameExprId, MemberAccessExprId
// CHECK:STDOUT: InstKind RefBinding: LetBindingPatternId, VarBindingPatternId
// CHECK:STDOUT: InstKind RefBindingPattern: VarBindingPatternId
// CHECK:STDOUT: InstKind RefParam: LetBindingPatternId
// CHECK:STDOUT: InstKind RefParamPattern: LetBindingPatternId
// CHECK:STDOUT: InstKind Return: FunctionDefinitionId
// CHECK:STDOUT: InstKind StructLiteral: StructLiteralId
// CHECK:STDOUT: InstKind VarPattern: VariablePatternId
// CHECK:STDOUT: InstKind VarStorage: VariablePatternId
// CHECK:STDOUT: InstKind WrapperBindingPattern: LetBindingPatternId
// CHECK:STDOUT:

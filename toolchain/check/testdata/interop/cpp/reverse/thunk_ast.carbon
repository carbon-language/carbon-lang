// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-cpp-ast
// SET-CHECK-SUBSET
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reverse/thunk_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reverse/thunk_ast.carbon
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon

// --- thunk_with_args_and_return.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F(i: i32) -> i32 {
// CHECK:STDOUT: | `-FunctionDecl {{0x[a-f0-9]+}} <thunk_with_args_and_return.carbon:[[@LINE-1]]:21> col:21 used F__cpp_thunk 'int (int)' inline
// CHECK:STDOUT: |   |-ParmVarDecl {{0x[a-f0-9]+}} <col:21> col:21 used 'int'
// CHECK:STDOUT: |   |-CompoundStmt {{0x[a-f0-9]+}} <col:21>
// CHECK:STDOUT: |   | |-DeclStmt {{0x[a-f0-9]+}} <col:21>
// CHECK:STDOUT: |   | | `-VarDecl {{0x[a-f0-9]+}} <col:21> col:21 used return_storage 'int' nrvo
// CHECK:STDOUT: |   | |-CallExpr {{0x[a-f0-9]+}} <col:21> 'void'
// CHECK:STDOUT: |   | | |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:21> 'void (*)(int &, int &)' <FunctionToPointerDecay>
// CHECK:STDOUT: |   | | | `-DeclRefExpr {{0x[a-f0-9]+}} <col:21> 'void (int &, int &)' Function {{0x[a-f0-9]+}} 'F__carbon_thunk' 'void (int &, int &)'
// CHECK:STDOUT: |   | | |-DeclRefExpr {{0x[a-f0-9]+}} <col:21> 'int' lvalue ParmVar {{0x[a-f0-9]+}} depth 0 index 0 'int'
// CHECK:STDOUT: |   | | `-DeclRefExpr {{0x[a-f0-9]+}} <col:21> 'int' lvalue Var {{0x[a-f0-9]+}} 'return_storage' 'int'
// CHECK:STDOUT: |   | `-ReturnStmt {{0x[a-f0-9]+}} <col:21> nrvo_candidate(Var {{0x[a-f0-9]+}} 'return_storage' 'int')
// CHECK:STDOUT: |   |   `-DeclRefExpr {{0x[a-f0-9]+}} <col:21> 'int' lvalue Var {{0x[a-f0-9]+}} 'return_storage' 'int'
// CHECK:STDOUT: |   |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: |   `-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <line:35:1, line:37:1> line:35:5 G 'int (int)'
// CHECK:STDOUT:   |-ParmVarDecl {{0x[a-f0-9]+}} <col:7, col:11> col:11 used i 'int'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:14, line:37:1>
// CHECK:STDOUT:     `-ReturnStmt {{0x[a-f0-9]+}} <line:36:3, col:21>
// CHECK:STDOUT:       `-CallExpr {{0x[a-f0-9]+}} <col:10, col:21> 'int'
// CHECK:STDOUT:         |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:10, col:18> 'int (*)(int)' <FunctionToPointerDecay>
// CHECK:STDOUT:         | `-DeclRefExpr {{0x[a-f0-9]+}} <col:10, col:18> 'int (int)' lvalue Function {{0x[a-f0-9]+}} 'F__cpp_thunk' 'int (int)'
// CHECK:STDOUT:         |   `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Carbon'
// CHECK:STDOUT:         `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:20> 'int' <LValueToRValue>
// CHECK:STDOUT:           `-DeclRefExpr {{0x[a-f0-9]+}} <col:20> 'int' lvalue ParmVar {{0x[a-f0-9]+}} 'i' 'int'
  return i;
}

inline Cpp '''
int G(int i) {
  return Carbon::F(i);
}
''';

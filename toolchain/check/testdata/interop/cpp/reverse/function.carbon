// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-cpp-ast
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reverse/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reverse/function.carbon
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other

// --- other.carbon

package Other;
fn F1() {}
// CHECK:STDOUT: |   `-FunctionDecl {{0x[a-f0-9]+}} <other.carbon:[[@LINE-1]]:9> col:9 used F1__cpp_thunk 'void ()' inline
// CHECK:STDOUT: |     |-CompoundStmt {{0x[a-f0-9]+}} <col:9>
// CHECK:STDOUT: |     | `-CallExpr {{0x[a-f0-9]+}} <col:9> 'void'
// CHECK:STDOUT: |     |   `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:9> 'void (*)()' <FunctionToPointerDecay>
// CHECK:STDOUT: |     |     `-DeclRefExpr {{0x[a-f0-9]+}} <col:9> 'void ()' Function {{0x[a-f0-9]+}} 'F1__carbon_thunk' 'void ()'
// CHECK:STDOUT: |     |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: |     `-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
fn F2() -> i32 { return 0; }
fn F3(_: i32) {}

fn HasGenericArg(T:! type, a: T) { a; }
fn HasDeducedArg[T:! type](a: T) { a; }

class C {
  fn Method[self: Self]() { self; }
}

// --- function.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <function.carbon:[[@LINE-1]]:1, line:30:1> line:6:6 G 'void ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:10, line:30:1>
// CHECK:STDOUT:     `-CallExpr {{0x[a-f0-9]+}} <line:29:3, col:21> 'void'
// CHECK:STDOUT:       `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:3, col:18> 'void (*)()' <FunctionToPointerDecay>
// CHECK:STDOUT:         `-DeclRefExpr {{0x[a-f0-9]+}} <col:3, col:18> 'void ()' lvalue Function {{0x[a-f0-9]+}} 'F1__cpp_thunk' 'void ()'
// CHECK:STDOUT:           `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Other'
// CHECK:STDOUT:             `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Carbon'
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other
// CHECK:STDOUT: |   `-FunctionDecl {{0x[a-f0-9]+}} <other.carbon:11:16> col:16 used F2__cpp_thunk 'int ()' inline
// CHECK:STDOUT: |     |-CompoundStmt {{0x[a-f0-9]+}} <col:16>
// CHECK:STDOUT: |     | |-DeclStmt {{0x[a-f0-9]+}} <col:16>
// CHECK:STDOUT: |     | | `-VarDecl {{0x[a-f0-9]+}} <col:16> col:16 used return_storage 'int' nrvo
// CHECK:STDOUT: |     | |-CallExpr {{0x[a-f0-9]+}} <col:16> 'void'
// CHECK:STDOUT: |     | | |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:16> 'void (*)(int &)' <FunctionToPointerDecay>
// CHECK:STDOUT: |     | | | `-DeclRefExpr {{0x[a-f0-9]+}} <col:16> 'void (int &)' Function {{0x[a-f0-9]+}} 'F2__carbon_thunk' 'void (int &)'
// CHECK:STDOUT: |     | | `-DeclRefExpr {{0x[a-f0-9]+}} <col:16> 'int' lvalue Var {{0x[a-f0-9]+}} 'return_storage' 'int'
// CHECK:STDOUT: |     | `-ReturnStmt {{0x[a-f0-9]+}} <col:16> nrvo_candidate(Var {{0x[a-f0-9]+}} 'return_storage' 'int')
// CHECK:STDOUT: |     |   `-DeclRefExpr {{0x[a-f0-9]+}} <col:16> 'int' lvalue Var {{0x[a-f0-9]+}} 'return_storage' 'int'
// CHECK:STDOUT: |     |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: |     `-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
  Carbon::Other::F1();
}
''';

// --- return_int.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <return_int.carbon:[[@LINE-1]]:1, line:28:1> line:6:5 G 'int ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:9, line:28:1>
// CHECK:STDOUT:     `-ReturnStmt {{0x[a-f0-9]+}} <line:27:3, col:28>
// CHECK:STDOUT:       `-CallExpr {{0x[a-f0-9]+}} <col:10, col:28> 'int'
// CHECK:STDOUT:         `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:10, col:25> 'int (*)()' <FunctionToPointerDecay>
// CHECK:STDOUT:           `-DeclRefExpr {{0x[a-f0-9]+}} <col:10, col:25> 'int ()' lvalue Function {{0x[a-f0-9]+}} 'F2__cpp_thunk' 'int ()'
// CHECK:STDOUT:             `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Other'
// CHECK:STDOUT:               `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Carbon'
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other
// CHECK:STDOUT: |   `-FunctionDecl {{0x[a-f0-9]+}} <other.carbon:12:15> col:15 used F3__cpp_thunk 'void (int)' inline
// CHECK:STDOUT: |     |-ParmVarDecl {{0x[a-f0-9]+}} <col:15> col:15 used 'int'
// CHECK:STDOUT: |     |-CompoundStmt {{0x[a-f0-9]+}} <col:15>
// CHECK:STDOUT: |     | `-CallExpr {{0x[a-f0-9]+}} <col:15> 'void'
// CHECK:STDOUT: |     |   |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:15> 'void (*)(int &)' <FunctionToPointerDecay>
// CHECK:STDOUT: |     |   | `-DeclRefExpr {{0x[a-f0-9]+}} <col:15> 'void (int &)' Function {{0x[a-f0-9]+}} 'F3__carbon_thunk' 'void (int &)'
// CHECK:STDOUT: |     |   `-DeclRefExpr {{0x[a-f0-9]+}} <col:15> 'int' lvalue ParmVar {{0x[a-f0-9]+}} depth 0 index 0 'int'
// CHECK:STDOUT: |     |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: |     `-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
  return Carbon::Other::F2();
}
''';

// --- args.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
void G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <args.carbon:[[@LINE-1]]:1, line:19:1> line:6:6 G 'void ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:10, line:19:1>
// CHECK:STDOUT:     `-CallExpr {{0x[a-f0-9]+}} <line:18:3, col:24> 'void'
// CHECK:STDOUT:       |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:3, col:18> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:STDOUT:       | `-DeclRefExpr {{0x[a-f0-9]+}} <col:3, col:18> 'void (int)' lvalue Function {{0x[a-f0-9]+}} 'F3__cpp_thunk' 'void (int)'
// CHECK:STDOUT:       |   `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Other'
// CHECK:STDOUT:       |     `-NestedNameSpecifier Namespace {{0x[a-f0-9]+}} 'Carbon'
// CHECK:STDOUT:       `-IntegerLiteral {{0x[a-f0-9]+}} <col:21> 'int' 123
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other
  Carbon::Other::F3(123);
}
''';

// --- fail_todo_generic.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:14:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with generic parameters` [SemanticsTodo]
// CHECK:STDERR: fn HasGenericArg(T:! type, a: T) { a; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <fail_todo_generic.carbon:[[@LINE-1]]:1, line:26:1> line:11:6 G 'void ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:10, line:26:1>
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other
  // CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+8]]:18: error: no member named 'HasGenericArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    25 |   Carbon::Other::HasGenericArg<int>(123);
  // CHECK:STDERR:       |                  ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_generic.carbon:[[@LINE+4]]:35: error: expected '(' for function-style cast or type construction [CppInteropParseError]
  // CHECK:STDERR:    25 |   Carbon::Other::HasGenericArg<int>(123);
  // CHECK:STDERR:       |                                ~~~^
  // CHECK:STDERR:
  Carbon::Other::HasGenericArg<int>(123);
}
''';

// --- fail_todo_deduced.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_deduced.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:15:1: error: semantics TODO: `unsupported: C++ calling a Carbon function with generic parameters` [SemanticsTodo]
// CHECK:STDERR: fn HasDeducedArg[T:! type](a: T) { a; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <fail_todo_deduced.carbon:[[@LINE-1]]:1, line:22:1> line:11:6 G 'void ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:10, line:22:1>
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon
// CHECK:STDOUT: | `-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Other
  // CHECK:STDERR: fail_todo_deduced.carbon:[[@LINE+4]]:18: error: no member named 'HasDeducedArg' in namespace 'Carbon::Other' [CppInteropParseError]
  // CHECK:STDERR:    21 |   Carbon::Other::HasDeducedArg(123);
  // CHECK:STDERR:       |                  ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Carbon::Other::HasDeducedArg(123);
}
''';

// --- fail_todo_method.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_method.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: other.carbon:18:3: error: semantics TODO: `unsupported: C++ calling a Carbon function with an implicit parameter` [SemanticsTodo]
// CHECK:STDERR:   fn Method[self: Self]() { self; }
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Other;
import Cpp inline '''
void G() {
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <fail_todo_method.carbon:[[@LINE-1]]:1, line:22:1> line:11:6 G 'void ()'
// CHECK:STDOUT:   `-CompoundStmt {{0x[a-f0-9]+}} <col:10, line:22:1>
// CHECK:STDOUT:     `-CallExpr {{0x[a-f0-9]+}} <line:21:3, col:29> '<dependent type>' contains-errors
// CHECK:STDOUT:       `-RecoveryExpr {{0x[a-f0-9]+}} <col:3, col:22> '<dependent type>' contains-errors lvalue
// CHECK:STDOUT:         `-CXXTemporaryObjectExpr {{0x[a-f0-9]+}} <col:3, col:20> 'Carbon::Other::C' 'void () noexcept' zeroing
  // CHECK:STDERR: fail_todo_method.carbon:[[@LINE+4]]:22: error: no member named 'Method' in 'Carbon::Other::C' [CppInteropParseError]
  // CHECK:STDERR:    21 |   Carbon::Other::C().Method();
  // CHECK:STDERR:       |   ~~~~~~~~~~~~~~~~~~ ^
  // CHECK:STDERR:
  Carbon::Other::C().Method();
}
''';

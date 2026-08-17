// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-cpp-ast --target=x86_64-linux-gnu
// SET-CHECK-SUBSET
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/thunk_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/thunk_ast.carbon
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-EmptyDecl {{0x[a-f0-9]+}} <<carbon Cpp imports>:1:1> col:1
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon external-linkage

// --- thunk_required.h

auto foo(short a) -> void;
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <./thunk_required.h:[[@LINE-1]]:1, col:22> col:6 in thunk_required.h hidden used foo 'auto (short) -> void' external-linkage
// CHECK:STDOUT: | `-ParmVarDecl {{0x[a-f0-9]+}} <col:10, col:16> col:16 in thunk_required.h hidden a 'short'

// --- import_thunk_required.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk_required.h";
// CHECK:STDOUT: |-ImportDecl {{0x[a-f0-9]+}} <import_thunk_required.carbon:[[@LINE-1]]:38> col:38 in thunk_required.h implicit thunk_required.h
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <./thunk_required.h:2:6> col:6 in import_thunk_required.carbon hidden foo__carbon_thunk 'void (short * _Nonnull)' inline internal-linkage
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <col:6> col:6 in import_thunk_required.carbon hidden used a 'short * _Nonnull':'short *'
// CHECK:STDOUT: | |-ReturnStmt {{0x[a-f0-9]+}} <col:6>
// CHECK:STDOUT: | | `-CallExpr {{0x[a-f0-9]+}} <col:6> 'void'
// CHECK:STDOUT: | |   |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'auto (*)(short) -> void' <FunctionToPointerDecay>
// CHECK:STDOUT: | |   | `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'auto (short) -> void' Function {{0x[a-f0-9]+}} 'foo' 'auto (short) -> void'
// CHECK:STDOUT: | |   `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'short' <LValueToRValue>
// CHECK:STDOUT: | |     `-UnaryOperator {{0x[a-f0-9]+}} <col:6> 'short' lvalue prefix '*' cannot overflow
// CHECK:STDOUT: | |       `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'short * _Nonnull':'short *' <LValueToRValue>
// CHECK:STDOUT: | |         `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'short * _Nonnull':'short *' lvalue ParmVar {{0x[a-f0-9]+}} 'a' 'short * _Nonnull':'short *'
// CHECK:STDOUT: | |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: | |-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | `-AsmLabelAttr {{0x[a-f0-9]+}} <col:6> Implicit "_Z3foos.carbon_thunk._"

fn F() {
  Cpp.foo(1 as i16);
}

// CHECK:STDOUT: `-ImportDecl {{0x[a-f0-9]+}} <import_thunk_required.carbon:[[@LINE+3]]:104> col:104 in import_thunk_required.carbon implicit import_thunk_required.carbon
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>
// CHECK:STDOUT: |-EmptyDecl {{0x[a-f0-9]+}} <<carbon Cpp imports>:1:1> col:1
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> Carbon external-linkage
// --- return_thunk_required.h

auto foo() -> short;
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <./return_thunk_required.h:[[@LINE-1]]:1, col:15> col:6 in return_thunk_required.h hidden used foo 'auto () -> short' external-linkage

// --- import_return_thunk_required.carbon

library "[[@TEST_NAME]]";

import Cpp library "return_thunk_required.h";
// CHECK:STDOUT: |-ImportDecl {{0x[a-f0-9]+}} <import_return_thunk_required.carbon:[[@LINE-1]]:45> col:45 in return_thunk_required.h implicit return_thunk_required.h
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <./return_thunk_required.h:2:6> col:6 in import_return_thunk_required.carbon hidden foo__carbon_thunk 'void (short * _Nonnull)' inline internal-linkage
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <col:6> col:6 in import_return_thunk_required.carbon hidden used return 'short * _Nonnull':'short *'
// CHECK:STDOUT: | |-CXXNewExpr {{0x[a-f0-9]+}} <col:6> 'short *' global Function {{0x[a-f0-9]+}} 'operator new' 'void *(__size_t, void *) noexcept'
// CHECK:STDOUT: | | |-CallExpr {{0x[a-f0-9]+}} <col:6> 'short'
// CHECK:STDOUT: | | | `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'auto (*)() -> short' <FunctionToPointerDecay>
// CHECK:STDOUT: | | |   `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'auto () -> short' Function {{0x[a-f0-9]+}} 'foo' 'auto () -> short'
// CHECK:STDOUT: | | `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'short * _Nonnull':'short *' lvalue ParmVar {{0x[a-f0-9]+}} 'return' 'short * _Nonnull':'short *'
// CHECK:STDOUT: | |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT: | |-InternalLinkageAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | `-AsmLabelAttr {{0x[a-f0-9]+}} <col:6> Implicit "_Z3foov.carbon_thunk."

fn F() -> i16 {
  return Cpp.foo();
}

// CHECK:STDOUT: `-ImportDecl {{0x[a-f0-9]+}} <import_return_thunk_required.carbon:[[@LINE+0]]:191> col:191 in import_return_thunk_required.carbon implicit import_return_thunk_required.carbon

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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/thunk_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/thunk_ast.carbon
// CHECK:STDOUT: TranslationUnitDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc>

// --- thunk_required.h

auto foo(short a) -> void;
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <./thunk_required.h:[[@LINE-1]]:1, col:22> col:6 used foo 'auto (short) -> void'
// CHECK:STDOUT: | `-ParmVarDecl {{0x[a-f0-9]+}} <col:10, col:16> col:16 a 'short'
// CHECK:STDOUT: |-NamespaceDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit std
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator new 'void *(__size_t)'
// CHECK:STDOUT: | |-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ReturnsNonNullAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | `-AllocSizeAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 1
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator new 'void *(__size_t, std::align_val_t)'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | |-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ReturnsNonNullAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | |-AllocSizeAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 1
// CHECK:STDOUT: | `-AllocAlignAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 2
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator new[] 'void *(__size_t)'
// CHECK:STDOUT: | |-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ReturnsNonNullAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | `-AllocSizeAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 1
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator new[] 'void *(__size_t, std::align_val_t)'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | |-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ReturnsNonNullAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit
// CHECK:STDOUT: | |-AllocSizeAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 1
// CHECK:STDOUT: | `-AllocAlignAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit 2
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator delete 'void (void *) noexcept'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator delete 'void (void *, std::align_val_t) noexcept'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator delete[] 'void (void *) noexcept'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit operator delete[] 'void (void *, std::align_val_t) noexcept'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'void *'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<invalid sloc>> <invalid sloc> implicit 'std::align_val_t'
// CHECK:STDOUT: | `-VisibilityAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit Default
// CHECK:STDOUT: |-FunctionDecl {{0x[a-f0-9]+}} <<carbon-internal>:8:1, line:14:1> line:8:7 operator new 'void *(unsigned long, void *) noexcept'
// CHECK:STDOUT: | |-ParmVarDecl {{0x[a-f0-9]+}} <<built-in>:173:23, col:37> <carbon-internal>:8:33 'unsigned long'
// CHECK:STDOUT: | `-ParmVarDecl {{0x[a-f0-9]+}} <col:35, col:39> col:40 'void *'
// CHECK:STDOUT: `-FunctionDecl {{0x[a-f0-9]+}} <./thunk_required.h:[[@LINE-51]]:6> col:6 foo__carbon_thunk 'void (short * _Nonnull)' extern
// CHECK:STDOUT:   |-ParmVarDecl {{0x[a-f0-9]+}} <col:6> col:6 used a 'short * _Nonnull':'short *'
// CHECK:STDOUT:   |-ReturnStmt {{0x[a-f0-9]+}} <col:6>
// CHECK:STDOUT:   | `-CallExpr {{0x[a-f0-9]+}} <col:6> 'void'
// CHECK:STDOUT:   |   |-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'auto (*)(short) -> void' <FunctionToPointerDecay>
// CHECK:STDOUT:   |   | `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'auto (short) -> void' Function {{0x[a-f0-9]+}} 'foo' 'auto (short) -> void'
// CHECK:STDOUT:   |   `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'short' <LValueToRValue>
// CHECK:STDOUT:   |     `-UnaryOperator {{0x[a-f0-9]+}} <col:6> 'short' lvalue prefix '*' cannot overflow
// CHECK:STDOUT:   |       `-ImplicitCastExpr {{0x[a-f0-9]+}} <col:6> 'short * _Nonnull':'short *' <LValueToRValue>
// CHECK:STDOUT:   |         `-DeclRefExpr {{0x[a-f0-9]+}} <col:6> 'short * _Nonnull':'short *' lvalue ParmVar {{0x[a-f0-9]+}} 'a' 'short * _Nonnull':'short *'
// CHECK:STDOUT:   |-AlwaysInlineAttr {{0x[a-f0-9]+}} <<invalid sloc>> Implicit always_inline
// CHECK:STDOUT:   `-AsmLabelAttr {{0x[a-f0-9]+}} <col:6> Implicit "_Z3foos.carbon_thunk"

// --- import_thunk_required.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk_required.h";

fn F() {
  Cpp.foo(1 as i16);
}

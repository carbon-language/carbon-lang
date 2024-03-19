#! /bin/bash
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: This is a short-term script to regenerate the test files for operators
# while we're in a period where we expect the tests to have substantial churn.
# Once the implementation of overloaded operators has stabilized, this script
# and its template files should be removed.

make_test() {
  HEADER="// This file was generated from $4. Run make_tests.sh to regenerate."
  sed "s,INTERFACE,$1,g; s,OP,$2,g; s,HEADER,$HEADER," "$4" > "$3.carbon"
}

make_unary_op_test() {
  make_test "$1" "$2" "$3" unary_op.carbon.tmpl
}

make_unary_stmt_test() {
  make_test "$1" "$2" "$3" unary_stmt.carbon.tmpl
}

make_binary_op_test() {
  make_test "$1" "$2" "$3" binary_op.carbon.tmpl
}

make_unary_op_test BitComplement '^' bit_complement
make_unary_op_test Negate '-' negate

make_unary_stmt_test Dec '--' dec
make_unary_stmt_test Inc '++' inc

make_binary_op_test Add '+' add
make_binary_op_test BitAnd '\&' bit_and
make_binary_op_test BitOr '|' bit_or
make_binary_op_test BitXor '^' bit_xor
make_binary_op_test Div '/' div
make_binary_op_test LeftShift '<<' left_shift
make_binary_op_test Mod '%' mod
make_binary_op_test Mul '*' mul
make_binary_op_test RightShift '>>' right_shift
make_binary_op_test Sub '-' sub

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is a comment.
    //    Comment with whitespace.
// class C; fn F[T:! A](a: T) -> i32 { return 86; }
//@dump-sem-ir-begin
//@dump-sem-ir-end
//@dump-sem-ir-
//no whitespace after double slash
/* invalid comment pattern. */
/// invalid comment pattern.
class C; // Comment after a statement/expression/declaration.

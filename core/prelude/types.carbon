// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types";

export import library "prelude/types/bool";

// TODO: Start importing `prelude/types/i32` here once we stop eagerly importing
// all `impl`s from all imported files. Currently, this introduces too much
// noise in the toolchain tests.
// import library "prelude/types/i32";

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(size: IntLiteral()) -> type = "int.make_type_signed";
fn UInt(size: IntLiteral()) -> type = "int.make_type_unsigned";
fn Float(size: IntLiteral()) -> type = "float.make_type";

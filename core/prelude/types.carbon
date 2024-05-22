// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types";

// TODO: Start importing `prelude/types/i32` here once we stop eagerly importing
// all `impl`s from all imported files. Currently, this introduces too much
// noise in the toolchain tests.
// import library "prelude/types/i32";

fn Int32() -> type = "int.make_type_32";
fn Int(size: i32) -> type = "int.make_type_signed";
fn UInt(size: i32) -> type = "int.make_type_unsigned";
fn Float(size: i32) -> type = "float.make_type";

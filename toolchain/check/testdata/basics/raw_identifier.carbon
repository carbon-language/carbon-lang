// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn A(n: i32) -> i32 {
  return r#n;
}

fn B(r#n: i32) -> i32 {
  return n;
}

fn C(r#if: i32) -> i32 {
  return r#if;
}

// CHECK:STDOUT: file "raw_identifier.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT:   %C: <function> = fn_decl @C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%if: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %if.ref: i32 = name_reference "if", %if
// CHECK:STDOUT:   return %if.ref
// CHECK:STDOUT: }

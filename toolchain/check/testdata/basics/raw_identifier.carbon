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

// CHECK:STDOUT: --- raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A, .B = %B, .C = %C} [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template]
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%if: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %if.ref: i32 = name_ref r#if, %if
// CHECK:STDOUT:   return %if.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

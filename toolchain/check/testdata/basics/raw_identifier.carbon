// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --disable-prelude-import --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn A(n: ()) -> () {
  return r#n;
}

fn B(r#n: ()) -> () {
  return n;
}

fn C(r#if: ()) -> () {
  return r#if;
}

// CHECK:STDOUT: --- raw_identifier.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {
// CHECK:STDOUT:     %.loc11_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_10.2: type = converted %.loc11_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc11_6.1: () = param n
// CHECK:STDOUT:     @A.%n: () = bind_name n, %n.loc11_6.1
// CHECK:STDOUT:     %.loc11_17.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_17.2: type = converted %.loc11_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var.loc11: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %.loc15_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc15_6.1: () = param n
// CHECK:STDOUT:     @B.%n: () = bind_name n, %n.loc15_6.1
// CHECK:STDOUT:     %.loc15_19.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19.2: type = converted %.loc15_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var.loc15: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc19_13.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_13.2: type = converted %.loc19_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %if.loc19_6.1: () = param r#if
// CHECK:STDOUT:     @C.%if: () = bind_name r#if, %if.loc19_6.1
// CHECK:STDOUT:     %.loc19_20.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_20.2: type = converted %.loc19_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var.loc19: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: () = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: () = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%if: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %if.ref: () = name_ref r#if, %if
// CHECK:STDOUT:   return %if.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

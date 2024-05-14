// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir %s
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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc11_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_10.2: type = converted %.loc11_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc11_6.1: () = param n
// CHECK:STDOUT:     @A.%n: () = bind_name n, %n.loc11_6.1
// CHECK:STDOUT:     %.loc11_17.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_17.2: type = converted %.loc11_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @A.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc15_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc15_6.1: () = param n
// CHECK:STDOUT:     @B.%n: () = bind_name n, %n.loc15_6.1
// CHECK:STDOUT:     %.loc15_19.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_19.2: type = converted %.loc15_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @B.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc19_13.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_13.2: type = converted %.loc19_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %if.loc19_6.1: () = param r#if
// CHECK:STDOUT:     @C.%if: () = bind_name r#if, %if.loc19_6.1
// CHECK:STDOUT:     %.loc19_20.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_20.2: type = converted %.loc19_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @C.%return: ref () = var <return slot>
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

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/raw_identifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/raw_identifier.carbon

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
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %.loc15_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_10.2: type = converted %.loc15_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc15_6.1: () = param n
// CHECK:STDOUT:     @A.%n: () = bind_name n, %n.loc15_6.1
// CHECK:STDOUT:     %.loc15_17.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_17.2: type = converted %.loc15_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @A.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: <type of B> = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %.loc19_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_12.2: type = converted %.loc19_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc19_6.1: () = param n
// CHECK:STDOUT:     @B.%n: () = bind_name n, %n.loc19_6.1
// CHECK:STDOUT:     %.loc19_19.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_19.2: type = converted %.loc19_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @B.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: <type of C> = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %.loc23_13.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc23_13.2: type = converted %.loc23_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %if.loc23_6.1: () = param r#if
// CHECK:STDOUT:     @C.%if: () = bind_name r#if, %if.loc23_6.1
// CHECK:STDOUT:     %.loc23_20.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc23_20.2: type = converted %.loc23_20.1, constants.%.1 [template = constants.%.1]
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

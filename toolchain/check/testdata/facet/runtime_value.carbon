// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/runtime_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/runtime_value.carbon

// --- facet_value_copy_from_reference.carbon
library "[[@TEST_NAME]]";

interface I {}

class C {
  var i: I;
}

fn F(c: C) {
  // Member access produces a reference, which is copied into the value
  // binding.
  //@dump-sem-ir-begin
  let a: I = c.i;
  //@dump-sem-ir-end
}

// --- fail_todo_facet_copy_narrowing_from_reference.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J {}

class C {
  var ij: I & J;
}

fn F(c: C) {
  // Member access produces a reference, which is copied into the value
  // binding.
  //
  // TODO: This copy should work, but the narrowing prevents it currently.
  //
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_facet_copy_narrowing_from_reference.carbon:[[@LINE+4]]:14: error: semantics TODO: `conversion of runtime facet value` [SemanticsTodo]
  // CHECK:STDERR:   let a: I = c.ij;
  // CHECK:STDERR:              ^~~~
  // CHECK:STDERR:
  let a: I = c.ij;
  //@dump-sem-ir-end
}

// --- fail_compound_member_lookup_in_runtime_facet_without_value.carbon
library "[[@TEST_NAME]]";

interface Z {
    let X:! type;
}

// CHECK:STDERR: fail_compound_member_lookup_in_runtime_facet_without_value.carbon:[[@LINE+4]]:15: error: semantics TODO: `associated value lookup on runtime facet value` [SemanticsTodo]
// CHECK:STDERR: fn F(T: Z, v: T.(Z.X));
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
fn F(T: Z, v: T.(Z.X));

// --- fail_member_lookup_in_runtime_facet_without_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// CHECK:STDERR: fail_member_lookup_in_runtime_facet_without_value.carbon:[[@LINE+4]]:15: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F(T: Z, v: T.X);
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
fn F(T: Z, v: T.X);

// --- fail_todo_compound_member_lookup_in_runtime_facet_with_value.carbon
library "[[@TEST_NAME]]";

interface Z {
    let X:! type;
}

// TODO: We should be able to get the value of `.X` from the `FacetType` of
// `T`.
//
// CHECK:STDERR: fail_todo_compound_member_lookup_in_runtime_facet_with_value.carbon:[[@LINE+4]]:29: error: semantics TODO: `conversion of runtime facet value` [SemanticsTodo]
// CHECK:STDERR: fn F(T: Z where .X = (), v: T.(Z.X));
// CHECK:STDERR:                             ^~~~~~~
// CHECK:STDERR:
fn F(T: Z where .X = (), v: T.(Z.X));

// --- fail_todo_member_lookup_in_runtime_facet_with_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// TODO: We should be able to get the value of `.X` from the `FacetType` of
// `T`.
//
// CHECK:STDERR: fail_todo_member_lookup_in_runtime_facet_with_value.carbon:[[@LINE+4]]:29: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F(T: Z where .X = (), v: T.X);
// CHECK:STDERR:                             ^~~
// CHECK:STDERR:
fn F(T: Z where .X = (), v: T.X);

// --- fail_todo_member_lookup_in_type_of_runtime_facet_with_value.carbon
library "[[@TEST_NAME]]";

interface Z {
  fn G();
}

fn F(T: Z) {
  // CHECK:STDERR: fail_todo_member_lookup_in_type_of_runtime_facet_with_value.carbon:[[@LINE+4]]:3: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// CHECK:STDOUT: --- facet_value_copy_from_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %I.type [concrete]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.09a = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %i.ref: %C.elem = name_ref i, @C.%.loc6 [concrete = @C.%.loc6]
// CHECK:STDOUT:   %.loc13_15.1: ref %I.type = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc13_15.2: %I.type = bind_value %.loc13_15.1
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %a: %I.type = value_binding a, %.loc13_15.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_facet_copy_narrowing_from_reference.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@I & @J> [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %facet_type [concrete]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.09a = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %ij.ref: %C.elem = name_ref ij, @C.%.loc7_9 [concrete = @C.%.loc7_9]
// CHECK:STDOUT:   %.loc21_15.1: ref %facet_type = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc21_15.2: %facet_type = bind_value %.loc21_15.1
// CHECK:STDOUT:   %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %a: %I.type = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

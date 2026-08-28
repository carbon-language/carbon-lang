// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/template.carbon

// --- deduced_or_explicit.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> T min(T a, T b) { return a < b ? a : b; }
''';

let a: i32 = 1;
let b: i32 = 2;
let min_implicit: i32 = Cpp.min(a, b);
let min_explicit: i32 = Cpp.min(i32, a, b);
let min_explicit_literal: i32 = Cpp.min(i32, 1, 2);

// --- template_template_arg.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename> struct TemplateTemplateX;
template<typename> struct TemplateTemplateY;
template<template<typename> typename... TT, typename... T> void template_template_fn(T ...a) {}
''';

fn Call() {
  let a: i32 = 0;
  let b: i32 = 0;

  // `a` and `b` are not classified as template arguments, so are passed as the
  // function parameter pack `a...`.
  Cpp.template_template_fn(a, b);

  // `Cpp.TemplateTemplateX` and `Cpp.TemplateTemplateY` are classified as template template arguments, so they
  // are passed as the template parameter pack `TT...`.
  Cpp.template_template_fn(Cpp.TemplateTemplateX, Cpp.TemplateTemplateY);

  // Pass some template arguments followed by some function arguments.
  Cpp.template_template_fn(Cpp.TemplateTemplateX, Cpp.TemplateTemplateY, a, b);
  Cpp.template_template_fn(Cpp.TemplateTemplateX, a, b, a);
}

// --- nontype_template_arg.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<int... I, typename... T> void nontype_f(T ...a) {}
template<int I = 0, int J = 0, typename = void, typename... T> void nontype_g(T ...a) {}
''';

fn Call() {
  let a: i32 = 0;
  let b: i32 = 0;

  Cpp.nontype_f(a, b);
  // TODO: There should be a way to indicate that the arguments are template
  // arguments, not function arguments, when calling `nontype_f`.
  Cpp.nontype_f(1, 2);

  // Two function arguments.
  Cpp.nontype_g(1, 2);
  // Three template arguments.
  Cpp.nontype_g(1, 2, i32);
  // Three template arguments and two function arguments.
  Cpp.nontype_g(1, 2, i32, 3, 4);
}

// --- fail_function_arg_not_template_arg.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<int N> void fn_arg_not_template_arg_f();
''';

fn Call() {
  // We treat this as a function argument, even though that makes the call fail.
  // CHECK:STDERR: fail_function_arg_not_template_arg.carbon:[[@LINE+4]]:3-31: error: no matching function for call to 'fn_arg_not_template_arg_f' [CppInteropParseError]
  // CHECK:STDERR: fail_function_arg_not_template_arg.carbon:[[@LINE-6]]:22-46: note: candidate function template not viable [CppInteropParseNote]
  // CHECK:STDERR: fail_function_arg_not_template_arg.carbon:[[@LINE-7]]:47-48: note: requires 0 arguments, but 1 was provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.fn_arg_not_template_arg_f(1);
}

// --- function_arg_not_template_arg_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<int N> void overloaded_f();
void overloaded_f(int n);
''';

fn Call() {
  Cpp.overloaded_f(1);
}

// --- overloaded_with_different_template_params.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename T> struct OverloadedTemplateX;
struct OverloadedNonTemplateY;

template<int N, typename T> void different_template_params_f();
template<typename T> void different_template_params_f();
template<template<typename> typename T> void different_template_params_f();
''';

fn Call() {
  Cpp.different_template_params_f(1, i32);
  Cpp.different_template_params_f(Cpp.OverloadedTemplateX);
  Cpp.different_template_params_f(Cpp.OverloadedNonTemplateY);
}

// --- only_call_templates_when_given_template_args.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct NonTemplate {};
struct Template {};

NonTemplate only_call_templates_f(unsigned char, unsigned char);
template<typename = void> Template only_call_templates_f(unsigned char, int = 0);
''';

// Prefers the non-template because it has a better match for the second parameter.
let x: Cpp.NonTemplate = Cpp.only_call_templates_f(1 as u8, 2 as u8);

// Calls the template; the non-template is not considered due to the template
// argument `i32`.
let y: Cpp.Template = Cpp.only_call_templates_f(i32, 1 as u8);

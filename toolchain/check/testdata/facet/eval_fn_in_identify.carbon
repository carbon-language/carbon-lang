// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/eval_fn_in_identify.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/eval_fn_in_identify.carbon

// --- fail_symbolic_call_in_identify_doesnt_find_self_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

eval fn LookupWithType(generic T: I) -> type {
  return T.(I.X);
}

class B {}
// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_doesnt_find_self_impl.carbon:[[@LINE+16]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_doesnt_find_self_impl.carbon:[[@LINE+8]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_symbolic_call_in_identify_with_generic_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class A {}
impl forall [T: type] T as I where .X = A and .Y = () {}

eval fn LookupWithType(generic T: type) -> type {
  return T.(I.X);
}

class B {}
// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_with_generic_impl.carbon:[[@LINE+16]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_with_generic_impl.carbon:[[@LINE+8]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_template_call_in_identify_doesnt_find_self_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

eval fn LookupWithType(template T: type) -> type {
  // This lookup is evaluated when .Self is replaced with B during identify. It
  // does a lookup for `B as I` which we are in the middle of declaring, so it
  // does not find the impl below. A use of `.X` directly in the impl
  // declaration would refer to its own impl, but in an `eval fn` we can't tell.
  // All templates of `LookupWithType(B)` must evaluate to the same thing, so
  // we can't rely on any state tracking that the call is happening inside the
  // declaration of an impl.
  return T.(I.X);
}

class B {}
// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_doesnt_find_self_impl.carbon:[[@LINE+16]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_doesnt_find_self_impl.carbon:[[@LINE+8]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_template_call_in_identify_with_generic_impl.carbon
library "[[@TEST_NAME]]";

interface I {
  let X: type;
  let Y: type;
}

class A {}
impl forall [T: type] T as I where .X = A and .Y = () {}

eval fn LookupWithType(template T: type) -> type {
  return T.(I.X);
}

class B {}
// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_with_generic_impl.carbon:[[@LINE+16]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_with_generic_impl.carbon:[[@LINE+8]]:36
// CHECK:STDERR: ->     │ impl B as I where .X = () and .Y = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                    ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                    ┗━━┫ error: compile-time call to `eval fn`
// CHECK:STDERR:        ·                                       ┃        with `.Self` dependency
// CHECK:STDERR:        ·                                       ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl B as I where .X = () and .Y = LookupWithType(.Self) {}

// --- fail_symbolic_call_in_identify_with_branch.carbon
library "[[@TEST_NAME]]";

interface I {
  let Cond: bool;
  let X: type;
}

eval fn LookupWithType(generic T: I) -> type {
  // There's no possible way to evaluate this if condition inside the
  // facet type of the `impl () as I` below. So if it was not an error
  // we end up crashing while trying to branch on an unknown symbolic
  // value.
  if (T.(I.Cond)) {
    return ();
  } else {
    return {};
  }
}

// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_with_branch.carbon:[[@LINE+18]]:43
// CHECK:STDERR: ->     │ impl () as I where .Cond = false and .X = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                           ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                           ┗━━┫ error: compile-time call to
// CHECK:STDERR:        ·                                              ┃        `eval fn` with `.Self`
// CHECK:STDERR:        ·                                              ┃        dependency
// CHECK:STDERR:        ·                                              ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_symbolic_call_in_identify_with_branch.carbon:[[@LINE+9]]:43
// CHECK:STDERR: ->     │ impl () as I where .Cond = false and .X = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                           ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                           ┗━━┫ error: compile-time call to
// CHECK:STDERR:        ·                                              ┃        `eval fn` with `.Self`
// CHECK:STDERR:        ·                                              ┃        dependency
// CHECK:STDERR:        ·                                              ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I where .Cond = false and .X = LookupWithType(.Self) {}

// --- fail_template_call_in_identify_with_branch.carbon
library "[[@TEST_NAME]]";

interface I {
  let Cond: bool;
  let X: type;
}

eval fn LookupWithType(template T: I) -> type {
  // There's no possible way to evaluate this if condition inside the
  // facet type of the `impl () as I` below. So if it was not an error
  // we end up crashing while trying to do impl lookup for `() as I`
  // in order to find the value of Cond for which there is no witness
  // table to find it in yet.
  if (T.(I.Cond)) {
    return ();
  } else {
    return {};
  }
}

// TODO: We get two errors here, one with frozen .Self and one thawed
// as they are different constants.
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_with_branch.carbon:[[@LINE+18]]:43
// CHECK:STDERR: ->     │ impl () as I where .Cond = false and .X = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                           ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                           ┗━━┫ error: compile-time call to
// CHECK:STDERR:        ·                                              ┃        `eval fn` with `.Self`
// CHECK:STDERR:        ·                                              ┃        dependency
// CHECK:STDERR:        ·                                              ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_template_call_in_identify_with_branch.carbon:[[@LINE+9]]:43
// CHECK:STDERR: ->     │ impl () as I where .Cond = false and .X = LookupWithType(.Self) {}
// CHECK:STDERR:        ·                                           ┳━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                           ┗━━┫ error: compile-time call to
// CHECK:STDERR:        ·                                              ┃        `eval fn` with `.Self`
// CHECK:STDERR:        ·                                              ┃        dependency
// CHECK:STDERR:        ·                                              ┃        [EvalFnPeriodSelfSymbolic]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as I where .Cond = false and .X = LookupWithType(.Self) {}

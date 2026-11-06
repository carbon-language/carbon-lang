// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/primitives.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/primitives.carbon

// --- todo_trivial_bool.carbon

library "[[@TEST_NAME]]";

fn Test() {
  bool as Core.Destroy;
  bool as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: bool = true;
  //@dump-sem-ir-end
}

// --- todo_trivial_bound_method.carbon

library "[[@TEST_NAME]]";

class C {
  fn Get(self) -> i32;
}

fn Make(c: C) -> typeof(c.Get);

fn Test(c: C) {
  typeof(c.Get) as Core.Destroy;
  typeof(c.Get) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(c.Get) = Make(c);
  //@dump-sem-ir-end
}

// --- todo_trivial_char_literal.carbon

library "[[@TEST_NAME]]";

fn Test() {
  // TODO: fix crash
  // Core.CharLiteral as Core.Destroy;
  // Core.CharLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // var unused x: Core.CharLiteral = 'c';
  //@dump-sem-ir-end
}

// --- todo_trivial_float.carbon

library "[[@TEST_NAME]]";

fn Test() {
  f64 as Core.Destroy;
  f64 as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: f64 = 0.0;
  //@dump-sem-ir-end
}

// --- todo_trivial_float_literal.carbon

library "[[@TEST_NAME]]";

fn Test() {
  // TODO: fix crash
  // Core.FloatLiteral as Core.Destroy;
  // Core.FloatLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // var unused x: Core.FloatLiteral = 0.0;
  //@dump-sem-ir-end
}

// --- todo_trivial_form.carbon

library "[[@TEST_NAME]]";

alias Form = typeof(form(var i32));
fn Make() -> Form;

fn Test() {
  Form as Core.Destroy;
  Form as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Form = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_form_with_nontrivial.carbon

library "[[@TEST_NAME]]";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused ref self) {}
  }
}

alias Form = typeof(form(var ImplsDestroy));
fn Make() -> Form;

fn Test() {
  Form as Core.Destroy;
  Form as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Form = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_function.carbon

library "[[@TEST_NAME]]";

fn G() {}

alias Fn = typeof(G);
fn Make() -> Fn;

fn Test() {
  Fn as Core.Destroy;
  Fn as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Fn = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_function_with_self.carbon

library "[[@TEST_NAME]]";

alias FnWithSelf = typeof((i32 as Core.Copy).Op);
fn Make() -> FnWithSelf;

fn Test() {
  // TODO: fix crash
  // FnWithSelf as Core.Destroy;
  // FnWithSelf as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // var unused x: FnWithSelf = Make();
  //@dump-sem-ir-end
}

// --- todo_trivial_int_literal.carbon

library "[[@TEST_NAME]]";

fn Test() {
  Core.IntLiteral as Core.Destroy;
  Core.IntLiteral as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: Core.IntLiteral = 0;
  //@dump-sem-ir-end
}

// --- todo_trivial_int.carbon

library "[[@TEST_NAME]]";

fn Test() {
  i32 as Core.Destroy;
  i32 as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: i32 = 0;
  //@dump-sem-ir-end
}

// --- type.carbon

library "[[@TEST_NAME]]";

fn Make() -> type;

fn Test() {
  type as Core.Destroy;
  type as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: type = Make();
  //@dump-sem-ir-end
}

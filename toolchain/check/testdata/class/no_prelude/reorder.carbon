// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/reorder.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/reorder.carbon

// --- reorder.carbon
library "[[@TEST_NAME]]";

class Class {
  fn G() -> () {
    return Class.F();
  }

  fn F() -> () {
    return ();
  }
}

// --- reorder_qualified.carbon
library "[[@TEST_NAME]]";

class A {
  class B {
    class C;

    fn BF();
    var b: ();
  }

  class B.C {
    class D {
      fn F();

      fn DF();
      var d: ();
    }

    fn D.DF() {
      // A, B, C, and D are complete here.
      var a: A = {.a = ()};
      var b: B = {.b = ()};
      var c: C = {.c = ()};
      var d: D = {.d = ()};

      // Unqualified lookup looks in all of them.
      AF();
      BF();
      CF();
      DF();
    }

    fn CF();
    var c: ();
  }

  fn AF();
  var a: ();
}

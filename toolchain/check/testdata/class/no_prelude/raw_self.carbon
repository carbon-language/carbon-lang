// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/raw_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/raw_self.carbon

// --- raw_self.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F[addr self: Self*](r#self: ());
  fn G[self: Self](r#self: ()) -> ((), ());
  var n: ();
}

fn Class.F[addr self: Self*](r#self: ()) {
  (*self).n = r#self;
}

fn Class.G[self: Self](r#self: ()) -> ((), ()) {
  return (self.n, r#self);
}

// --- raw_self_type.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() {
    var r#Self: Self*;
    var p: Self* = r#Self;
  }
}

class MemberNamedSelf {
  class r#Self {}

  fn F(x: Self, y: r#Self);
}

fn MemberNamedSelf.F(x: Self, y: r#Self) {}

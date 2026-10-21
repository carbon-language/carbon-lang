// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/default_values.carbon

// --- generic_fn_specific_param.carbon
library "[[@TEST_NAME]]";

class D {}
impl D as Core.Copy {
  fn Op(unused self: D) -> D { return {} as D; }
}

class E {}
impl E as Core.Copy {
  fn Op(unused self: E) -> E { return {} as E; }
}

class C(T: Core.Copy & Core.Destroy) {
  var t: T;
  var d: D;
  impl as Core.Copy {
    fn Op(self: Self) -> Self {
      let c: Self = {.t = self.t, .d = self.d};
      return c;
    }
  }
}

fn Make(generic T: Core.Copy & Core.Destroy, t: T, d: D = {} as D) -> C(T) {
  let c: C(T) = {.t = t, .d = d};
  return c;
}

fn G() {
  let _: C(E) = Make(E, {} as E, {} as D);
  //@dump-sem-ir-begin
  let _: C(E) = Make(E, {} as E);
  //@dump-sem-ir-end
}

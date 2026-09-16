// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/class.carbon

// --- init_class.carbon
library "[[@TEST_NAME]]";

class B {
  impl as Core.Copy {
    fn Op(unused self) -> Self {
      return {};
    }
  }
}

class A(template T: type) {
  fn Make(v: T) -> A(T) {
    return {
      .field = v,
    };
  }

  var field: T;
}

fn Run() {
  A(B).Make({});
}

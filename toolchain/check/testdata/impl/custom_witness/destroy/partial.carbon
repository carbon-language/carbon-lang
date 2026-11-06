// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy/partial.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class ImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class AlsoImplsDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

// --- trivial_empty_partial_class.carbon

library "[[@TEST_NAME]]";

base class Base {}

fn Test() {
  var unused x: partial Base = {};
}

// --- trivial_nonempty_partial_class.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

base class Base {
  var a: A;
  var b: B;
}

fn Test() {
  var unused x: partial Base = {.a = {}, .b = {}};
}

// --- nontrivial_member.carbon

library "[[@TEST_NAME]]";

import library "types";

base class Base {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Base = {.x = {}, .y = {}};
  //@dump-sem-ir-end
}

// --- nontrivial_impls_destroy.carbon

library "[[@TEST_NAME]]";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Base = {};
  //@dump-sem-ir-end
}

// --- trivial_abstract.carbon

library "[[@TEST_NAME]]";

abstract class Abstract {}

fn Test() {
  var unused x: partial Abstract = {};
}

// --- nontrivial_abstract.carbon

library "[[@TEST_NAME]]";

import library "types";

abstract class Abstract {
  var x: ImplsDestroy;
  var y: AlsoImplsDestroy;
}

fn Test() {
  //@dump-sem-ir-begin
  var unused x: partial Abstract = {.x = {}, .y = {}};
  //@dump-sem-ir-end
}

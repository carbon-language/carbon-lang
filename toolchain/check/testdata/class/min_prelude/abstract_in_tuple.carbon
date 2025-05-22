// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/abstract_in_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/abstract_in_tuple.carbon

// --- fail_abstract_field.carbon
library "[[@TEST_NAME]]";

abstract class Abstract1 {}

class Contains {
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE+7]]:10: error: field has abstract type `(Abstract1,)` [AbstractTypeInFieldDecl]
  // CHECK:STDERR:   var a: (Abstract1,);
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract1 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: (Abstract1,);
}

// --- fail_abstract_var.carbon
library "[[@TEST_NAME]]";

abstract class Abstract2 {}

fn Var() {
  // CHECK:STDERR: fail_abstract_var.carbon:[[@LINE+7]]:10: error: binding pattern has abstract type `(Abstract2,)` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v: (Abstract2,);
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_var.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract2 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: (Abstract2,);
}

// --- abstract_let.carbon
library "[[@TEST_NAME]]";

abstract class Abstract3 {
}

fn F(a: Abstract3) {
  //@dump-sem-ir-begin
  let l: (Abstract3,) = (a,);
  //@dump-sem-ir-end
}

// --- fail_abstract_twice.carbon
library "[[@TEST_NAME]]";

abstract class Abstract4 {}
abstract class Abstract5 {}

fn Var2() {
  // CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE+7]]:11: error: binding pattern has abstract type `(Abstract4, Abstract5)` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v2: (Abstract4, Abstract5);
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE-7]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract4 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v2: (Abstract4, Abstract5);
}

// --- fail_abstract_first.carbon
library "[[@TEST_NAME]]";

abstract class Abstract6 {}

fn Var3() {
  // CHECK:STDERR: fail_abstract_first.carbon:[[@LINE+7]]:11: error: binding pattern has abstract type `(Abstract6, {})` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v3: (Abstract6, {});
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_first.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract6 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v3: (Abstract6, {});
}

// --- fail_abstract_second.carbon
library "[[@TEST_NAME]]";

abstract class Abstract7 {}

fn Var4() {
  // CHECK:STDERR: fail_abstract_second.carbon:[[@LINE+7]]:11: error: binding pattern has abstract type `({}, Abstract7)` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v4: ({}, Abstract7);
  // CHECK:STDERR:           ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_second.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract7 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v4: ({}, Abstract7);
}

// --- lib.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}

// --- fail_import.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Var5() {
  // CHECK:STDERR: fail_import.carbon:[[@LINE+8]]:11: error: binding pattern has abstract type `(Abstract,)` in `var` pattern [AbstractTypeInVarPattern]
  // CHECK:STDERR:   var v5: (Abstract,);
  // CHECK:STDERR:           ^~~~~~~~~~~
  // CHECK:STDERR: fail_import.carbon:[[@LINE-6]]:1: in import [InImport]
  // CHECK:STDERR: lib.carbon:3:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v5: (Abstract,);
}

// CHECK:STDOUT: --- abstract_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract3: type = class_type @Abstract3 [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.fa1: type = tuple_type (%Abstract3) [concrete]
// CHECK:STDOUT:   %pattern_type.3cf: type = pattern_type %tuple.type.fa1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %Abstract3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %l.patt: %pattern_type.3cf = binding_pattern l [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %Abstract3 = name_ref a, %a
// CHECK:STDOUT:   %.loc8_28: %tuple.type.fa1 = tuple_literal (%a.ref)
// CHECK:STDOUT:   %.loc8_21.1: type = splice_block %.loc8_21.3 [concrete = constants.%tuple.type.fa1] {
// CHECK:STDOUT:     %Abstract3.ref.loc8: type = name_ref Abstract3, file.%Abstract3.decl [concrete = constants.%Abstract3]
// CHECK:STDOUT:     %.loc8_21.2: %tuple.type.85c = tuple_literal (%Abstract3.ref.loc8)
// CHECK:STDOUT:     %.loc8_21.3: type = converted %.loc8_21.2, constants.%tuple.type.fa1 [concrete = constants.%tuple.type.fa1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %l: %tuple.type.fa1 = bind_name l, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/abstract_in_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/abstract_in_struct.carbon

// --- fail_abstract_field.carbon
library "[[@TEST_NAME]]";

abstract class Abstract1 {}

class Contains {
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE+7]]:10: error: field has abstract type `{.m1: Abstract1}` [AbstractTypeInFieldDecl]
  // CHECK:STDERR:   var a: {.m1: Abstract1};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract_field.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR: abstract class Abstract1 {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: {.m1: Abstract1};
}

// --- fail_abstract_var.carbon
library "[[@TEST_NAME]]";

abstract class Abstract2 {}

// CHECK:STDERR: fail_abstract_var.carbon:[[@LINE+7]]:8: error: binding pattern has abstract type `{.m2: Abstract2}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v: {.m2: Abstract2};
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_var.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract2 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v: {.m2: Abstract2};

// --- abstract_let.carbon
library "[[@TEST_NAME]]";

abstract class Abstract3 {
}

fn F(a: Abstract3) {
  // TODO: This should probably require `partial`.
  //@dump-sem-ir-begin
  let l: {.m3: Abstract3} = {.m3 = a};
  //@dump-sem-ir-end
}

// --- fail_abstract_twice.carbon
library "[[@TEST_NAME]]";

abstract class Abstract4 {}
abstract class Abstract5 {}

// CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.m4: Abstract4, .m5: Abstract5}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v2: {.m4: Abstract4, .m5: Abstract5};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_twice.carbon:[[@LINE-6]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract4 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v2: {.m4: Abstract4, .m5: Abstract5};

// --- fail_abstract_first.carbon
library "[[@TEST_NAME]]";

abstract class Abstract6 {}

// CHECK:STDERR: fail_abstract_first.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.m6: Abstract6, .c1: ()}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v3: {.m6: Abstract6, .c1: ()};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_first.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract6 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v3: {.m6: Abstract6, .c1: ()};

// --- fail_abstract_second.carbon
library "[[@TEST_NAME]]";

abstract class Abstract7 {}

// CHECK:STDERR: fail_abstract_second.carbon:[[@LINE+7]]:9: error: binding pattern has abstract type `{.c2: (), .m7: Abstract7}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v4: {.c2: (), .m7: Abstract7};
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_second.carbon:[[@LINE-5]]:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract7 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v4: {.c2: (), .m7: Abstract7};

// --- lib.carbon
library "[[@TEST_NAME]]";

abstract class Abstract {}

// --- fail_import.carbon
library "[[@TEST_NAME]]";

import library "lib";

// CHECK:STDERR: fail_import.carbon:[[@LINE+8]]:9: error: binding pattern has abstract type `{.m: Abstract}` in `var` pattern [AbstractTypeInVarPattern]
// CHECK:STDERR: var v5: {.m: Abstract};
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_import.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: lib.carbon:3:1: note: uses class that was declared abstract here [ClassAbstractHere]
// CHECK:STDERR: abstract class Abstract {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v5: {.m: Abstract};

// CHECK:STDOUT: --- abstract_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract3: type = class_type @Abstract3 [concrete]
// CHECK:STDOUT:   %struct_type.m3.fe4: type = struct_type {.m3: %Abstract3} [concrete]
// CHECK:STDOUT:   %pattern_type.a22: type = pattern_type %struct_type.m3.fe4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %Abstract3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %l.patt: %pattern_type.a22 = binding_pattern l [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %Abstract3 = name_ref a, %a
// CHECK:STDOUT:   %.loc9_37: %struct_type.m3.fe4 = struct_literal (%a.ref)
// CHECK:STDOUT:   %.loc9_25: type = splice_block %struct_type.m3 [concrete = constants.%struct_type.m3.fe4] {
// CHECK:STDOUT:     %Abstract3.ref.loc9: type = name_ref Abstract3, file.%Abstract3.decl [concrete = constants.%Abstract3]
// CHECK:STDOUT:     %struct_type.m3: type = struct_type {.m3: %Abstract3} [concrete = constants.%struct_type.m3.fe4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %l: %struct_type.m3.fe4 = bind_name l, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/var.carbon

// --- global.carbon

library "[[@TEST_NAME]]";

var a: () = ();

fn F() -> () {
  //@dump-sem-ir-begin
  // TODO: Decide if this should be allowed, and if so, whether it should be
  // spelled as
  //
  //   alias b = ref a;
  alias b = a;
  //@dump-sem-ir-end
  return b;
}

// --- fail_local.carbon

library "[[@TEST_NAME]]";

fn F() -> () {
  var a: () = ();
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_local.carbon:[[@LINE+4]]:13: error: alias refers to a runtime value [AliasRequiresConstantValue]
  // CHECK:STDERR:   alias b = a;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  alias b = a;
  //@dump-sem-ir-end
  return b;
}

// CHECK:STDOUT: --- global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %b: ref %empty_tuple.type = alias_binding b, %a.ref [concrete = file.%a.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %b: ref %empty_tuple.type = alias_binding b, %a.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

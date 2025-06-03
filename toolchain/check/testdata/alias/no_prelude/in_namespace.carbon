// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/in_namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/in_namespace.carbon

// --- in_namespace.carbon

library "[[@TEST_NAME]]";

class C { var v: (); }

//@dump-sem-ir-begin
namespace NS;
alias NS.a = C;
//@dump-sem-ir-end

let b: NS.a = {.v = ()};

fn F() -> NS.a {
  return {.v = ()};
}

// --- fail_local_in_namespace.carbon

library "[[@TEST_NAME]]";

namespace NS;

fn F() -> {} {
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+8]]:9: error: name `NS` not found [NameNotFound]
  // CHECK:STDERR:   alias NS.a = {};
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+4]]:16: error: alias initializer must be a name reference [AliasRequiresNameRef]
  // CHECK:STDERR:   alias NS.a = {};
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  alias NS.a = {};
  // CHECK:STDERR: fail_local_in_namespace.carbon:[[@LINE+4]]:10: error: member name `a` not found in `NS` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return NS.a;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return NS.a;
}

// CHECK:STDOUT: --- in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

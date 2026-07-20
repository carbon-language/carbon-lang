// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if/lifetime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if/lifetime.carbon

// --- condition_lifetime.carbon
library "[[@TEST_NAME]]";

class A {
  fn F() -> bool;
}

fn Make() -> A;

fn G();

fn F() {
  //@dump-sem-ir-begin
  // The temporary A object is destroyed before the branch.
  if (Make().F()) {
    G();
  }
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- condition_lifetime.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc14_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_12.1: ref %A = temporary_storage
// CHECK:STDOUT:   %Make.call: init %A to %.loc14_12.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc14_12.2: ref %A = temporary %.loc14_12.1, %Make.call
// CHECK:STDOUT:   %F.ref: %A.F.type = name_ref F, @A.%A.F.decl [concrete = constants.%A.F]
// CHECK:STDOUT:   %A.F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc14_17.1: bool = value_of_initializer %A.F.call
// CHECK:STDOUT:   %.loc14_17.2: bool = converted %A.F.call, %.loc14_17.1
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc14_12.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc14_12.2)
// CHECK:STDOUT:   if %.loc14_17.2 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_12.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_12.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

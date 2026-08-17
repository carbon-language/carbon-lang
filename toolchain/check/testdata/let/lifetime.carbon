// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/lifetime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/lifetime.carbon

// --- extend.carbon
library "[[@TEST_NAME]]";

class A { fn Clone() -> Self; }

fn G();

fn F() {
  //@dump-sem-ir-begin
  // All of these A temporaries get lifetime-extended.
  let unused a: A = ({} as A, {} as A).1;
  let unused b: A = ({} as A).Clone();
  G();
  // Four A objects are destroyed here.
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %A.Clone.type: type = fn_type @A.Clone [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.Clone: %A.Clone.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.777: type = tuple_type (%A, %A) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.777 = tuple_value (%A.val, %A.val) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.367: ref %A = temporary invalid, %A.val [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.9ef = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc11_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.367, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_23.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc10_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc10_23.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc10_23.3: init %A to %.loc10_23.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_25.1: init %A = converted %.loc10_23.1, %.loc10_23.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_32.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc10_37: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc10_32.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc10_32.3: init %A to %.loc10_32.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_34.1: init %A = converted %.loc10_32.1, %.loc10_32.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_38.1: %tuple.type.777 = tuple_literal (%.loc10_25.1, %.loc10_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_25.2: ref %A = temporary %.loc10_23.2, %.loc10_25.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc10_25.3: %A = acquire_value %.loc10_25.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_34.2: ref %A = temporary %.loc10_32.2, %.loc10_34.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %.loc10_34.3: %A = acquire_value %.loc10_34.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %tuple: %tuple.type.777 = tuple_value (%.loc10_25.3, %.loc10_34.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc10_38.2: %tuple.type.777 = converted %.loc10_38.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem1: %A = tuple_access %.loc10_38.2, element1 [concrete = constants.%A.val]
// CHECK:STDOUT:   %A.ref.loc10_17: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: %A = wrapper_binding a, %tuple.elem1
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_23.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %A.ref.loc11_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc11_23.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc11_23.3: init %A to %.loc11_23.2 = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc11_25.1: init %A = converted %.loc11_23.1, %.loc11_23.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc11_25.2: ref %A = temporary %.loc11_23.2, %.loc11_25.1 [concrete = constants.%.367]
// CHECK:STDOUT:   %Clone.ref: %A.Clone.type = name_ref Clone, @A.%A.Clone.decl [concrete = constants.%A.Clone]
// CHECK:STDOUT:   %.loc11_37.1: ref %A = temporary_storage
// CHECK:STDOUT:   %A.Clone.call: init %A to %.loc11_37.1 = call %Clone.ref()
// CHECK:STDOUT:   %.loc11_37.2: ref %A = temporary %.loc11_37.1, %A.Clone.call
// CHECK:STDOUT:   %.loc11_37.3: %A = acquire_value %.loc11_37.2
// CHECK:STDOUT:   %A.ref.loc11_17: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %b: %A = wrapper_binding b, %.loc11_37.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.9ef = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref()
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc11_37.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc11_37: init %empty_tuple.type = call %Destroy.Op.bound(%.loc11_37.2)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_25: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_34: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_25: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.367)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_37.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_37.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

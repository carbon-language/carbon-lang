// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/import.carbon

// --- library.carbon

library "[[@TEST_NAME]]";

fn F() -> array(i32, 42);

// --- user.carbon

import library "library";

fn G(n: i32) -> i32 {
  //@dump-sem-ir-begin
  return F()[n];
  //@dump-sem-ir-end
}

// --- fail_todo_symbolic_decl.carbon

library "[[@TEST_NAME]]";

interface I {
  let val:! array(i32, 1);
}

class C {}

// CHECK:STDERR: fail_todo_symbolic_decl.carbon:[[@LINE+4]]:26: error: expression is runtime; expected constant [EvalRequiresConstantValue]
// CHECK:STDERR: impl C as I where .val = (1,) {}
// CHECK:STDERR:                          ^~~~
// CHECK:STDERR:
impl C as I where .val = (1,) {}

// --- fail_todo_import_symbolic_decl.carbon

library "[[@TEST_NAME]]";
import library "symbolic_decl";

fn F() -> array(i32, 1) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_symbolic_decl.carbon:[[@LINE+4]]:11: error: cannot convert type `C` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   return (C as I).val;
  // CHECK:STDERR:           ^~~~~~
  // CHECK:STDERR:
  return (C as I).val;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %ptr.830: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.886: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.923: %T.as.Destroy.impl.Op.type.886 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//library, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6_12.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type = call %F.ref() to %.loc6_12.1
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc6_12.2: ref %array_type = temporary %.loc6_12.1, %F.call
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6_15.1: ref %i32 = array_index %.loc6_12.2, %n.ref
// CHECK:STDOUT:   %.loc6_15.2: %i32 = bind_value %.loc6_15.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc6_12.2, constants.%T.as.Destroy.impl.Op.923
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc6_12.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.830 = addr_of %.loc6_12.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc6_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_symbolic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//symbolic_decl, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//symbolic_decl, C, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   %val.ref: <error> = name_ref val, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

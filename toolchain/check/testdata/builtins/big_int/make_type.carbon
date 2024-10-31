// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/big_int/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/big_int/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn BigInt() -> type = "big_int.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

var i: BigInt();

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BigInt.type: type = fn_type @BigInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BigInt: %BigInt.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .BigInt = %BigInt.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %BigInt.decl: %BigInt.type = fn_decl @BigInt [template = constants.%BigInt] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BigInt() -> type = "big_int.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %BigInt.type: type = fn_type @BigInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %BigInt: %BigInt.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %BigInt.type = import_ref Main//types, inst+7, loaded [template = constants.%BigInt]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .BigInt = imports.%import_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .i = %i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %BigInt.ref: %BigInt.type = name_ref BigInt, imports.%import_ref [template = constants.%BigInt]
// CHECK:STDOUT:   %big_int.make_type: init type = call %BigInt.ref() [template = Core.BigInt]
// CHECK:STDOUT:   %.loc6_15.1: type = value_of_initializer %big_int.make_type [template = Core.BigInt]
// CHECK:STDOUT:   %.loc6_15.2: type = converted %big_int.make_type, %.loc6_15.1 [template = Core.BigInt]
// CHECK:STDOUT:   %i.var: ref Core.BigInt = var i
// CHECK:STDOUT:   %i: ref Core.BigInt = bind_name i, %i.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BigInt() -> type = "big_int.make_type";
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_var_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_var_in_type.carbon

var x: type = i32;
// CHECK:STDERR: fail_var_in_type.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn Six() -> x { return 6; }

// CHECK:STDOUT: --- fail_var_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.de9: <witness> = impl_witness imports.%Copy.impl_witness_table.40f [concrete]
// CHECK:STDOUT:   %Copy.facet.c80: %Copy.type = facet_value type, (%Copy.impl_witness.de9) [concrete]
// CHECK:STDOUT:   %.98f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c80 [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.type: type = fn_type @type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op: %type.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.bound: <bound method> = bound_method %i32, %type.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Six.type: type = fn_type @Six [concrete]
// CHECK:STDOUT:   %Six: %Six.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.f97: %type.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc40_31, loaded [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.40f = impl_witness_table (%Core.import_ref.f97), @type.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Six = %Six.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.98f = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.98f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %x: ref type = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   %Six.decl: %Six.type = fn_decl @Six [concrete = constants.%Six] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.ref: ref type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:     %.loc20: type = bind_value %x.ref
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.98f = impl_witness_access constants.%Copy.impl_witness.de9, element0 [concrete = constants.%type.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %i32, %impl.elem0 [concrete = constants.%type.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %type.as.Copy.impl.Op.call: init type = call %bound_method(%i32) [concrete = constants.%i32]
// CHECK:STDOUT:   assign file.%x.var, %type.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

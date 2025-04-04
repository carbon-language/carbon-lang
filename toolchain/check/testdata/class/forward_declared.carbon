// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/forward_declared.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/forward_declared.carbon

class Class;

fn F(p: Class*) -> Class* { return p; }

// CHECK:STDOUT: --- forward_declared.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc13_20: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc13_25: type = ptr_type %Class.ref.loc13_20 [concrete = constants.%ptr]
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %ptr.loc13_14 [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Class.ref.loc13_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc13_14: type = ptr_type %Class.ref.loc13_9 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %ptr) -> %ptr {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

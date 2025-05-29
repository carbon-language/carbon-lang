// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_import_type_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_import_type_error.carbon

// --- fail_implicit.carbon

package Implicit;

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:12: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var a_ref: x;
// CHECK:STDERR:            ^
// CHECK:STDERR:
var a_ref: x;
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:18: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var b_ref: {.a = x};
// CHECK:STDERR:                  ^
// CHECK:STDERR:
var b_ref: {.a = x};
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:13: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var c_ref: (x,);
// CHECK:STDERR:             ^
// CHECK:STDERR:
var c_ref: (x,);
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:12: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var d_ref: x*;
// CHECK:STDERR:            ^
// CHECK:STDERR:
var d_ref: x*;

// --- implicit.impl.carbon

impl package Implicit;

// In each of these cases, name lookup should succeed, but they should all have
// an error type.
var a: i32 = a_ref;
var b: i32 = b_ref;
var c: i32 = c_ref;
var d: i32 = d_ref;

// CHECK:STDOUT: --- fail_implicit.carbon
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
// CHECK:STDOUT:     .x = <poisoned>
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .c_ref = %c_ref
// CHECK:STDOUT:     .d_ref = %d_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: <error> = binding_pattern a_ref [concrete]
// CHECK:STDOUT:     %a_ref.var_patt: <error> = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref <error> = var %a_ref.var_patt [concrete = <error>]
// CHECK:STDOUT:   %x.ref.loc8: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   %a_ref: <error> = bind_name a_ref, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ref.patt: <error> = binding_pattern b_ref [concrete]
// CHECK:STDOUT:     %b_ref.var_patt: <error> = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref.var: ref <error> = var %b_ref.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %x.ref.loc13: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13: <error> = struct_literal (%x.ref.loc13)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref: <error> = bind_name b_ref, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_ref.patt: <error> = binding_pattern c_ref [concrete]
// CHECK:STDOUT:     %c_ref.var_patt: <error> = var_pattern %c_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ref.var: ref <error> = var %c_ref.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.2: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %x.ref.loc18: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc18: <error> = tuple_literal (%x.ref.loc18)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ref: <error> = bind_name c_ref, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d_ref.patt: <error> = binding_pattern d_ref [concrete]
// CHECK:STDOUT:     %d_ref.var_patt: <error> = var_pattern %d_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d_ref.var: ref <error> = var %d_ref.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:     %x.ref.loc23: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:     %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d_ref: <error> = bind_name d_ref, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: <error> = import_ref Implicit//default, a_ref, loaded [concrete = <error>]
// CHECK:STDOUT:   %Implicit.b_ref: <error> = import_ref Implicit//default, b_ref, loaded [concrete = <error>]
// CHECK:STDOUT:   %Implicit.c_ref: <error> = import_ref Implicit//default, c_ref, loaded [concrete = <error>]
// CHECK:STDOUT:   %Implicit.d_ref: <error> = import_ref Implicit//default, d_ref, loaded [concrete = <error>]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .c_ref = imports.%Implicit.c_ref
// CHECK:STDOUT:     .d_ref = imports.%Implicit.d_ref
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %i32.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7ce = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %i32.loc9 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: <error> = name_ref a_ref, imports.%Implicit.a_ref [concrete = <error>]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %b_ref.ref: <error> = name_ref b_ref, imports.%Implicit.b_ref [concrete = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %c_ref.ref: <error> = name_ref c_ref, imports.%Implicit.c_ref [concrete = <error>]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %d_ref.ref: <error> = name_ref d_ref, imports.%Implicit.d_ref [concrete = <error>]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_deref_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_deref_type.carbon

// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+7]]:8: error: cannot dereference operand of non-pointer type `type` [DerefOfNonPointer]
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+4]]:8: note: to form a pointer type, write the `*` after the pointee type [DerefOfType]
// CHECK:STDERR: var p: *i32;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var p: *i32;
// CHECK:STDERR: fail_deref_type.carbon:[[@LINE+4]]:12: error: cannot apply `->` operator to non-pointer type `type` [ArrowOperatorOfNonPointer]
// CHECK:STDERR: var p2: i32->foo;
// CHECK:STDERR:            ^~
// CHECK:STDERR:
var p2: i32->foo;

// CHECK:STDOUT: --- fail_deref_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:     .p2 = %p2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: <error> = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: <error> = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref <error> = var %p.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: ref <error> = deref %i32.loc22 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref <error> = ref_binding p, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p2.patt: <error> = ref_binding_pattern p2 [concrete]
// CHECK:STDOUT:     %p2.var_patt: <error> = var_pattern %p2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p2.var: ref <error> = var %p2.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.2: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc27: ref <error> = deref %i32.loc27 [concrete = <error>]
// CHECK:STDOUT:     %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p2: ref <error> = ref_binding p2, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:

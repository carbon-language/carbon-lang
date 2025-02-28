// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/redeclare.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/redeclare.carbon

// --- valid.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type);

class Generic(T:! type) {
}

// --- fail_mismatch_param_list.carbon

library "[[@TEST_NAME]]";

class A;
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: class A(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE-4]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class A(T:! type) {}

// --- fail_mismatch_implicit_param_list.carbon

library "[[@TEST_NAME]]";

class B(N:! i32);
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE+7]]:1: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
// CHECK:STDERR: class B[T:! type](N:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE-4]]:1: note: previously declared without implicit parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class B(N:! i32);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class B[T:! type](N:! T) {}

// --- fail_mismatch_param_count.carbon

library "[[@TEST_NAME]]";

class C(T:! type);
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: class C(T:! type, U:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class C(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T:! type, U:! i32) {}

// --- fail_mismatch_param_type.carbon

library "[[@TEST_NAME]]";

class D(T:! type);
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE+7]]:9: error: type `i32` of parameter 1 in redeclaration differs from previous parameter type `type` [RedeclParamDiffersType]
// CHECK:STDERR: class D(T:! i32) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class D(T:! type);
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR:
class D(T:! i32) {}

// --- fail_mismatch_param_name.carbon

library "[[@TEST_NAME]]";

class E(T:! type);
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE+7]]:9: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class E(U:! type) {}
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class E(T:! type);
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR:
class E(U:! type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl.loc4: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6, runtime_param<none> [symbolic = constants.%T.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param.loc4: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, %T.param.loc4 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6, runtime_param<none> [symbolic = constants.%T.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param.loc6: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T, 0, %T.param.loc6 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.e41: type = class_type @.1, @.1(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %T.patt.loc12_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_9.1, runtime_param<none> [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc12_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%T.loc12_9.1: type) {
// CHECK:STDOUT:   %T.loc12_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.e41
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T) {
// CHECK:STDOUT:   %T.loc12_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_implicit_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %N.f22: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.51c: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.943: type = class_type @.1, @.1(%T, %N.f22) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: %B.type = class_decl @B [concrete = constants.%B.generic] {
// CHECK:STDOUT:     %N.patt.loc4_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_9.1, runtime_param<none> [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %T.patt.loc12_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_9.1, runtime_param<none> [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc12_19.1: @.1.%T.loc12_9.2 (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc12_19.2 (constants.%N.patt.51c)]
// CHECK:STDOUT:     %N.param_patt: @.1.%T.loc12_9.2 (%T) = value_param_pattern %N.patt.loc12_19.1, runtime_param<none> [symbolic = %N.patt.loc12_19.2 (constants.%N.patt.51c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc12_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: @.1.%T.loc12_9.2 (%T) = value_param runtime_param<none>
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_9.1 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:     %N.loc12_19.1: @.1.%T.loc12_9.2 (%T) = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc12_19.2 (constants.%N.f22)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(%N.loc4_9.1: %i32) {
// CHECK:STDOUT:   %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc4_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%T.loc12_9.1: type, %N.loc12_19.1: @.1.%T.loc12_9.2 (%T)) {
// CHECK:STDOUT:   %T.loc12_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc12_19.2: %T = bind_symbolic_name N, 1 [symbolic = %N.loc12_19.2 (constants.%N.f22)]
// CHECK:STDOUT:   %N.patt.loc12_19.2: %T = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc12_19.2 (constants.%N.patt.51c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.943
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_9.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc4_9.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%N.f22) {
// CHECK:STDOUT:   %T.loc12_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_9.2 => constants.%T
// CHECK:STDOUT:   %N.loc12_19.2 => constants.%N.f22
// CHECK:STDOUT:   %N.patt.loc12_19.2 => constants.%N.f22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %U: %i32 = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: %i32 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c28: type = class_type @.1, @.1(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<none> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %T.patt.loc12_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_9.1, runtime_param<none> [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc12_19.1: %i32 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_19.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %i32 = value_param_pattern %U.patt.loc12_19.1, runtime_param<none> [symbolic = %U.patt.loc12_19.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc12_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_19.1: %i32 = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc12_19.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%T.loc12_9.1: type, %U.loc12_19.1: %i32) {
// CHECK:STDOUT:   %T.loc12_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc12_19.2: %i32 = bind_symbolic_name U, 1 [symbolic = %U.loc12_19.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_19.2: %i32 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc12_19.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.c28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc12_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_9.2 => constants.%T
// CHECK:STDOUT:   %U.loc12_19.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_19.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %T.51e: %i32 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.8e2: %i32 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.f5e: type = class_type @.1, @.1(%T.51e) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [concrete = constants.%D.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<none> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %T.patt.loc12_9.1: %i32 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt.8e2)]
// CHECK:STDOUT:     %T.param_patt: %i32 = value_param_pattern %T.patt.loc12_9.1, runtime_param<none> [symbolic = %T.patt.loc12_9.2 (constants.%T.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_9.1: %i32 = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_9.2 (constants.%T.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%T.loc12_9.1: %i32) {
// CHECK:STDOUT:   %T.loc12_9.2: %i32 = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T.51e)]
// CHECK:STDOUT:   %T.patt.loc12_9.2: %i32 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_9.2 (constants.%T.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.f5e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%T.51e) {
// CHECK:STDOUT:   %T.loc12_9.2 => constants.%T.51e
// CHECK:STDOUT:   %T.patt.loc12_9.2 => constants.%T.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %E.type: type = generic_class_type @E [concrete]
// CHECK:STDOUT:   %E.generic: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [concrete]
// CHECK:STDOUT:   %.generic: %.type = struct_value () [concrete]
// CHECK:STDOUT:   %.e41: type = class_type @.1, @.1(%U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl: %E.type = class_decl @E [concrete = constants.%E.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<none> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [concrete = constants.%.generic] {
// CHECK:STDOUT:     %U.patt.loc12_9.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_9.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc12_9.1, runtime_param<none> [symbolic = %U.patt.loc12_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %U.loc12_9.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc12_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @.1(%U.loc12_9.1: type) {
// CHECK:STDOUT:   %U.loc12_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_9.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_9.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_9.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%.e41
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%U) {
// CHECK:STDOUT:   %U.loc12_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:

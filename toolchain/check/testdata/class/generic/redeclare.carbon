// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

class A {}

class B(N:! A);
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE+7]]:1: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
// CHECK:STDERR: class B[T:! type](N:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE-4]]:1: note: previously declared without implicit parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class B(N:! A);
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
class B[T:! type](N:! T) {}

// --- fail_mismatch_param_count.carbon

library "[[@TEST_NAME]]";

class A {}

class C(T:! type);
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: class C(T:! type, U:! A) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class C(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T:! type, U:! A) {}

// --- fail_mismatch_param_type.carbon

library "[[@TEST_NAME]]";

class A {}

class D(T:! type);
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE+7]]:9: error: type `<pattern for A>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: class D(T:! A) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class D(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class D(T:! A) {}

// --- fail_mismatch_param_name.carbon

library "[[@TEST_NAME]]";

class E(T:! type);
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE+7]]:9: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class E(U:! type) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class E(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class E(U:! type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.466: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A.2 [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.130: type = class_type @A.2, @A.2(%T) [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc4: type = class_decl @A.1 [concrete = constants.%A.466] {} {}
// CHECK:STDOUT:   %A.decl.loc12: %A.type = class_decl @A.2 [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc12_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.1;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A.2(%T.loc12_9.1: type) {
// CHECK:STDOUT:   %T.loc12_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.130
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%T) {
// CHECK:STDOUT:   %T.loc12_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_implicit_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.9e6: %A = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %B.type.844c0f.1: type = generic_class_type @B.1 [concrete]
// CHECK:STDOUT:   %B.generic.ba299b.1: %B.type.844c0f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %N.f22: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %B.type.844c0f.2: type = generic_class_type @B.2 [concrete]
// CHECK:STDOUT:   %B.generic.ba299b.2: %B.type.844c0f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %B.828: type = class_type @B.2, @B.2(%T, %N.f22) [symbolic]
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
// CHECK:STDOUT:     .B = %B.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc6: %B.type.844c0f.1 = class_decl @B.1 [concrete = constants.%B.generic.ba299b.1] {
// CHECK:STDOUT:     %N.patt: %pattern_type.c10 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %N.loc6_9.1: %A = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.9e6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc14: %B.type.844c0f.2 = class_decl @B.2 [concrete = constants.%B.generic.ba299b.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @B.2.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_9.1 [symbolic = %T.loc14_9.2 (constants.%T)]
// CHECK:STDOUT:     %N.loc14_19.1: @B.2.%T.loc14_9.2 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc14_19.2 (constants.%N.f22)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.1(%N.loc6_9.1: %A) {
// CHECK:STDOUT:   %N.loc6_9.2: %A = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.9e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.2(%T.loc14_9.1: type, %N.loc14_19.1: @B.2.%T.loc14_9.2 (%T)) {
// CHECK:STDOUT:   %T.loc14_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T)]
// CHECK:STDOUT:   %N.loc14_19.2: @B.2.%T.loc14_9.2 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc14_19.2 (constants.%N.f22)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc14_9.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.828
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%N.9e6) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N.9e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.2(constants.%T, constants.%N.f22) {
// CHECK:STDOUT:   %T.loc14_9.2 => constants.%T
// CHECK:STDOUT:   %N.loc14_19.2 => constants.%N.f22
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type.e6e560.1: type = generic_class_type @C.1 [concrete]
// CHECK:STDOUT:   %C.generic.965b12.1: %C.type.e6e560.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U: %A = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %C.type.e6e560.2: type = generic_class_type @C.2 [concrete]
// CHECK:STDOUT:   %C.generic.965b12.2: %C.type.e6e560.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.3d8: type = class_type @C.2, @C.2(%T, %U) [symbolic]
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
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %C.decl.loc6: %C.type.e6e560.1 = class_decl @C.1 [concrete = constants.%C.generic.965b12.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc14: %C.type.e6e560.2 = class_decl @C.2 [concrete = constants.%C.generic.965b12.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.c10 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T)]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %U.loc14_19.1: %A = bind_symbolic_name U, 1 [symbolic = %U.loc14_19.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.1(%T.loc6_9.1: type) {
// CHECK:STDOUT:   %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.2(%T.loc14_9.1: type, %U.loc14_19.1: %A) {
// CHECK:STDOUT:   %T.loc14_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc14_19.2: %A = bind_symbolic_name U, 1 [symbolic = %U.loc14_19.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3d8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.1(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc14_9.2 => constants.%T
// CHECK:STDOUT:   %U.loc14_19.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D.type.bbd080.1: type = generic_class_type @D.1 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.1: %D.type.bbd080.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.9e6: %A = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %D.type.bbd080.2: type = generic_class_type @D.2 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.2: %D.type.bbd080.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.384: type = class_type @D.2, @D.2(%T.9e6) [symbolic]
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
// CHECK:STDOUT:     .D = %D.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %D.decl.loc6: %D.type.bbd080.1 = class_decl @D.1 [concrete = constants.%D.generic.4e2319.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc14: %D.type.bbd080.2 = class_decl @D.2 [concrete = constants.%D.generic.4e2319.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c10 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %T.loc14_9.1: %A = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T.9e6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.1(%T.loc6_9.1: type) {
// CHECK:STDOUT:   %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.2(%T.loc14_9.1: %A) {
// CHECK:STDOUT:   %T.loc14_9.2: %A = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.2 (constants.%T.9e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.384
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.1(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc6_9.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.2(constants.%T.9e6) {
// CHECK:STDOUT:   %T.loc14_9.2 => constants.%T.9e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %E.type.b0f8dc.1: type = generic_class_type @E.1 [concrete]
// CHECK:STDOUT:   %E.generic.f281ba.1: %E.type.b0f8dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %E.type.b0f8dc.2: type = generic_class_type @E.2 [concrete]
// CHECK:STDOUT:   %E.generic.f281ba.2: %E.type.b0f8dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %E.ec9c10.2: type = class_type @E.2, @E.2(%U) [symbolic]
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
// CHECK:STDOUT:     .E = %E.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl.loc4: %E.type.b0f8dc.1 = class_decl @E.1 [concrete = constants.%E.generic.f281ba.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl.loc12: %E.type.b0f8dc.2 = class_decl @E.2 [concrete = constants.%E.generic.f281ba.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc12_9.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.1(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.2(%U.loc12_9.1: type) {
// CHECK:STDOUT:   %U.loc12_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_9.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E.ec9c10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.2(constants.%U) {
// CHECK:STDOUT:   %U.loc12_9.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:

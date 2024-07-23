// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/syntactic_merge_literal.carbon
// CHECK:STDERR: fail_int_mismatch.carbon: ERROR: Main//default previously provided by `int_match.carbon`.
// CHECK:STDERR:

// --- int_match.carbon

class C(a: i32) {}
class D(b: C(1_000));
class D(b: C(1_000)) {}

// --- fail_int_mismatch.carbon

class C(a: i32) {}
class D(b: C(1000));
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE+6]]:14: ERROR: Redeclaration syntax differs here.
// CHECK:STDERR: class D(b: C(1_000)) {}
// CHECK:STDERR:              ^~~~~
// CHECK:STDERR: fail_int_mismatch.carbon:[[@LINE-4]]:14: Comparing with previous declaration here.
// CHECK:STDERR: class D(b: C(1000));
// CHECK:STDERR:              ^~~~
class D(b: C(1_000)) {}

// CHECK:STDOUT: --- int_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1000 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, invalid(%.3) [template]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.1: %D.type = struct_value () [template]
// CHECK:STDOUT:   %D.2: type = class_type @D [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc2_12.2: type = converted %int.make_type_32, %.loc2_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_9.1: i32 = param a
// CHECK:STDOUT:     %a.loc2_9.2: i32 = bind_name a, %a.loc2_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc3: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %C.ref.loc3: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc3_14: i32 = int_literal 1000 [template = constants.%.3]
// CHECK:STDOUT:     %.loc3_13: init type = call %C.ref.loc3(%.loc3_14) [template = constants.%C.3]
// CHECK:STDOUT:     %.loc3_19.1: type = value_of_initializer %.loc3_13 [template = constants.%C.3]
// CHECK:STDOUT:     %.loc3_19.2: type = converted %.loc3_13, %.loc3_19.1 [template = constants.%C.3]
// CHECK:STDOUT:     %b.loc3_9.1: %C.3 = param b
// CHECK:STDOUT:     %b.loc3_9.2: %C.3 = bind_name b, %b.loc3_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc4: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %C.ref.loc4: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc4_14: i32 = int_literal 1000 [template = constants.%.3]
// CHECK:STDOUT:     %.loc4_13: init type = call %C.ref.loc4(%.loc4_14) [template = constants.%C.3]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %.loc4_13 [template = constants.%C.3]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %.loc4_13, %.loc4_19.1 [template = constants.%C.3]
// CHECK:STDOUT:     %b.loc4_9.1: %C.3 = param b
// CHECK:STDOUT:     %b.loc4_9.2: %C.3 = bind_name b, %b.loc4_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific invalid(constants.%.3);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1000 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, invalid(%.3) [template]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [template]
// CHECK:STDOUT:   %D.1: %D.type = struct_value () [template]
// CHECK:STDOUT:   %D.2: type = class_type @D [template]
// CHECK:STDOUT:   %.type: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = class_type @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc2_12.2: type = converted %int.make_type_32, %.loc2_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_9.1: i32 = param a
// CHECK:STDOUT:     %a.loc2_9.2: i32 = bind_name a, %a.loc2_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = class_decl @D [template = constants.%D.1] {
// CHECK:STDOUT:     %C.ref.loc3: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc3_14: i32 = int_literal 1000 [template = constants.%.3]
// CHECK:STDOUT:     %.loc3_13: init type = call %C.ref.loc3(%.loc3_14) [template = constants.%C.3]
// CHECK:STDOUT:     %.loc3_18.1: type = value_of_initializer %.loc3_13 [template = constants.%C.3]
// CHECK:STDOUT:     %.loc3_18.2: type = converted %.loc3_13, %.loc3_18.1 [template = constants.%C.3]
// CHECK:STDOUT:     %b.loc3_9.1: %C.3 = param b
// CHECK:STDOUT:     %b.loc3_9.2: %C.3 = bind_name b, %b.loc3_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = class_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %C.ref.loc10: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %.loc10_14: i32 = int_literal 1000 [template = constants.%.3]
// CHECK:STDOUT:     %.loc10_13: init type = call %C.ref.loc10(%.loc10_14) [template = constants.%C.3]
// CHECK:STDOUT:     %.loc10_19.1: type = value_of_initializer %.loc10_13 [template = constants.%C.3]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_13, %.loc10_19.1 [template = constants.%C.3]
// CHECK:STDOUT:     %b.loc10_9.1: %C.3 = param b
// CHECK:STDOUT:     %b.loc10_9.2: %C.3 = bind_name b, %b.loc10_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific invalid(constants.%.3);
// CHECK:STDOUT:

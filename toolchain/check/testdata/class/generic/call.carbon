// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/call.carbon

// --- call.carbon

library "call";

class Class(T:! type, N:! i32) {}

var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);

// --- fail_too_few.carbon

library "too_few";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+7]]:8: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var a: Class(i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_few.carbon:[[@LINE-5]]:1: Calling function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*);

// --- fail_too_many.carbon

library "too_many";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+7]]:8: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var a: Class(i32*, 1, 2);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_many.carbon:[[@LINE-5]]:1: Calling function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*, 1, 2);

// --- fail_no_conversion.carbon

library "no_conversion";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+6]]:8: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE-5]]:1: Initializing parameter 1 of function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var a: Class(5, i32*);

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%.3, %.4) [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%.1, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
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
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = @Class.%N (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_17.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_17.2: type = converted %int.make_type_32.loc6, %.loc6_17.1 [template = i32]
// CHECK:STDOUT:   %.loc6_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_20: i32 = int_literal 5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_13: init type = call %Class.ref.loc6(%.loc6_17.3, %.loc6_20) [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc6_21.1: type = value_of_initializer %.loc6_13 [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc6_21.2: type = converted %.loc6_13, %.loc6_21.1 [template = constants.%Class.3]
// CHECK:STDOUT:   %a.var: ref %Class.3 = var a
// CHECK:STDOUT:   %a: ref %Class.3 = bind_name a, %a.var
// CHECK:STDOUT:   %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc9_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.1: type = converted %.loc9_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_13.2: init type = call %Class.ref.loc9(%.loc9_13.1, %.loc9_18) [template = constants.%Class.4]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %.loc9_13.2 [template = constants.%Class.4]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %.loc9_13.2, %.loc9_19.1 [template = constants.%Class.4]
// CHECK:STDOUT:   %b.var: ref %Class.4 = var b
// CHECK:STDOUT:   %b: ref %Class.4 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type, file.%N.loc4_23.2: i32) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.3, constants.%.4) {
// CHECK:STDOUT:   %T => constants.%.3
// CHECK:STDOUT:   %N => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.1, constants.%.6) {
// CHECK:STDOUT:   %T => constants.%.1
// CHECK:STDOUT:   %N => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
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
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = @Class.%N (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32.loc13, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_13: init type = call %Class.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc13_18.1: type = value_of_initializer %.loc13_13 [template = <error>]
// CHECK:STDOUT:   %.loc13_18.2: type = converted %.loc13_13, %.loc13_18.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type, file.%N.loc4_23.2: i32) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
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
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = @Class.%N (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32.loc13, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_23: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_13: init type = call %Class.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc13_24.1: type = value_of_initializer %.loc13_13 [template = <error>]
// CHECK:STDOUT:   %.loc13_24.2: type = converted %.loc13_13, %.loc13_24.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type, file.%N.loc4_23.2: i32) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
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
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = @Class.%N (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_20.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_20.2: type = converted %int.make_type_32.loc12, %.loc12_20.1 [template = i32]
// CHECK:STDOUT:   %.loc12_20.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_13: init type = call %Class.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %.loc12_13 [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %.loc12_13, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type, file.%N.loc4_23.2: i32) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

// --- call_in_nested_return_type.carbon

class Outer(T:! type) {
  class Inner(U:! type) {
    fn A() -> Outer(T) {
      return {};
    }
    fn B() -> Outer(U) {
      return {};
    }
    fn C() -> Inner(T) {
      return {};
    }
    fn D() -> Inner(U) {
      return {};
    }
  }
}

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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N
// CHECK:STDOUT:     %N.loc4: i32 = bind_symbolic_name N 1, %N.param [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc6_17.2: type = converted %int.make_type_32, %.loc6_17.1 [template = i32]
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
// CHECK:STDOUT: generic class @Class(%T.loc4: type, %N.loc4: i32) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 1 [symbolic = %N.1 (constants.%N)]
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
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %N.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.3, constants.%.4) {
// CHECK:STDOUT:   %T.1 => constants.%.3
// CHECK:STDOUT:   %N.1 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.1, constants.%.6) {
// CHECK:STDOUT:   %T.1 => constants.%.1
// CHECK:STDOUT:   %N.1 => constants.%.6
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N
// CHECK:STDOUT:     %N.loc4: i32 = bind_symbolic_name N 1, %N.param [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_13: init type = call %Class.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc13_18.1: type = value_of_initializer %.loc13_13 [template = <error>]
// CHECK:STDOUT:   %.loc13_18.2: type = converted %.loc13_13, %.loc13_18.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4: type, %N.loc4: i32) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 1 [symbolic = %N.1 (constants.%N)]
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
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %N.1 => constants.%N
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N
// CHECK:STDOUT:     %N.loc4: i32 = bind_symbolic_name N 1, %N.param [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32, %.loc13_17.1 [template = i32]
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
// CHECK:STDOUT: generic class @Class(%T.loc4: type, %N.loc4: i32) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 1 [symbolic = %N.1 (constants.%N)]
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
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %N.1 => constants.%N
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N
// CHECK:STDOUT:     %N.loc4: i32 = bind_symbolic_name N 1, %N.param [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_20.2: type = converted %int.make_type_32, %.loc12_20.1 [template = i32]
// CHECK:STDOUT:   %.loc12_20.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_13: init type = call %Class.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %.loc12_13 [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %.loc12_13, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4: type, %N.loc4: i32) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 1 [symbolic = %N.1 (constants.%N)]
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
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %N.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Outer.1: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner [template]
// CHECK:STDOUT:   %Inner.1: %Inner.type = struct_value () [template]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %A.type.1: type = fn_type @A, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %A.1: %A.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.3: type = class_type @Outer, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %B.type.1: type = fn_type @B, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %B.1: %B.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.3: type = class_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %C.type.1: type = fn_type @C, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %C.1: %C.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.4: type = class_type @Inner, @Inner(%U) [symbolic]
// CHECK:STDOUT:   %D.type.1: type = fn_type @D, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %D.1: %D.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %struct.1: %Outer.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.2: %Outer.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.3: %Inner.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.type.2: type = fn_type @A, @Inner(%U, %U) [symbolic]
// CHECK:STDOUT:   %A.2: %A.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %B.type.2: type = fn_type @B, @Inner(%U, %U) [symbolic]
// CHECK:STDOUT:   %B.2: %B.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.type.2: type = fn_type @C, @Inner(%U, %U) [symbolic]
// CHECK:STDOUT:   %C.2: %C.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %D.type.2: type = fn_type @D, @Inner(%U, %U) [symbolic]
// CHECK:STDOUT:   %D.2: %D.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.4: %Inner.4 = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [template = constants.%Outer.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc2: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc2: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: %Inner.type = class_decl @Inner [template = constants.%Inner.1] {
// CHECK:STDOUT:       %U.patt: type = symbolic_binding_pattern U 1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = param U
// CHECK:STDOUT:       %U.loc3: type = bind_symbolic_name U 1, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc2: type, %U.loc3: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A.type: type = fn_type @A, @Inner(%T, %U.1) [symbolic = %A.type (constants.%A.type.1)]
// CHECK:STDOUT:   %A: @Inner.%A.type (%A.type.1) = struct_value () [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %B.type: type = fn_type @B, @Inner(%T, %U.1) [symbolic = %B.type (constants.%B.type.1)]
// CHECK:STDOUT:   %B: @Inner.%B.type (%B.type.1) = struct_value () [symbolic = %B (constants.%B.1)]
// CHECK:STDOUT:   %C.type: type = fn_type @C, @Inner(%T, %U.1) [symbolic = %C.type (constants.%C.type.1)]
// CHECK:STDOUT:   %C: @Inner.%C.type (%C.type.1) = struct_value () [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %D.type: type = fn_type @D, @Inner(%T, %U.1) [symbolic = %D.type (constants.%D.type.1)]
// CHECK:STDOUT:   %D: @Inner.%D.type (%D.type.1) = struct_value () [symbolic = %D (constants.%D.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %A.decl: @Inner.%A.type (%A.type.1) = fn_decl @A [symbolic = @Inner.%A (constants.%A.1)] {} {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc4_20: init type = call %Outer.ref(%T.ref) [symbolic = %Outer (constants.%Outer.2)]
// CHECK:STDOUT:       %.loc4_22.1: type = value_of_initializer %.loc4_20 [symbolic = %Outer (constants.%Outer.2)]
// CHECK:STDOUT:       %.loc4_22.2: type = converted %.loc4_20, %.loc4_22.1 [symbolic = %Outer (constants.%Outer.2)]
// CHECK:STDOUT:       %return: ref @A.%Outer (%Outer.2) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.decl: @Inner.%B.type (%B.type.1) = fn_decl @B [symbolic = @Inner.%B (constants.%B.1)] {} {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %.loc7_20: init type = call %Outer.ref(%U.ref) [symbolic = %Outer (constants.%Outer.3)]
// CHECK:STDOUT:       %.loc7_22.1: type = value_of_initializer %.loc7_20 [symbolic = %Outer (constants.%Outer.3)]
// CHECK:STDOUT:       %.loc7_22.2: type = converted %.loc7_20, %.loc7_22.1 [symbolic = %Outer (constants.%Outer.3)]
// CHECK:STDOUT:       %return: ref @B.%Outer (%Outer.3) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: @Inner.%C.type (%C.type.1) = fn_decl @C [symbolic = @Inner.%C (constants.%C.1)] {} {
// CHECK:STDOUT:       %Inner.ref: %Inner.type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc10_20: init type = call %Inner.ref(%T.ref) [symbolic = %Inner (constants.%Inner.3)]
// CHECK:STDOUT:       %.loc10_22.1: type = value_of_initializer %.loc10_20 [symbolic = %Inner (constants.%Inner.3)]
// CHECK:STDOUT:       %.loc10_22.2: type = converted %.loc10_20, %.loc10_22.1 [symbolic = %Inner (constants.%Inner.3)]
// CHECK:STDOUT:       %return: ref @C.%Inner (%Inner.3) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.decl: @Inner.%D.type (%D.type.1) = fn_decl @D [symbolic = @Inner.%D (constants.%D.1)] {} {
// CHECK:STDOUT:       %Inner.ref: %Inner.type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner.1]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %.loc13_20: init type = call %Inner.ref(%U.ref) [symbolic = %Inner (constants.%Inner.4)]
// CHECK:STDOUT:       %.loc13_22.1: type = value_of_initializer %.loc13_20 [symbolic = %Inner (constants.%Inner.4)]
// CHECK:STDOUT:       %.loc13_22.2: type = converted %.loc13_20, %.loc13_22.1 [symbolic = %Inner (constants.%Inner.4)]
// CHECK:STDOUT:       %return: ref @D.%Inner (%Inner.4) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.2
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(@Outer.%T.loc2: type, @Inner.%U.loc3: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @A.%Outer (%Outer.2) = struct_value () [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @A.%Outer (%Outer.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_15.1: %.2 = struct_literal ()
// CHECK:STDOUT:     %.loc5_15.2: init @A.%Outer (%Outer.2) = class_init (), %return [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:     %.loc5_16: init @A.%Outer (%Outer.2) = converted %.loc5_15.1, %.loc5_15.2 [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:     return %.loc5_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B(@Outer.%T.loc2: type, @Inner.%U.loc3: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%U) [symbolic = %Outer (constants.%Outer.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @B.%Outer (%Outer.3) = struct_value () [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @B.%Outer (%Outer.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_15.1: %.2 = struct_literal ()
// CHECK:STDOUT:     %.loc8_15.2: init @B.%Outer (%Outer.3) = class_init (), %return [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:     %.loc8_16: init @B.%Outer (%Outer.3) = converted %.loc8_15.1, %.loc8_15.2 [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:     return %.loc8_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C(@Outer.%T.loc2: type, @Inner.%U.loc3: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @C.%Inner (%Inner.3) = struct_value () [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @C.%Inner (%Inner.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15.1: %.2 = struct_literal ()
// CHECK:STDOUT:     %.loc11_15.2: init @C.%Inner (%Inner.3) = class_init (), %return [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:     %.loc11_16: init @C.%Inner (%Inner.3) = converted %.loc11_15.1, %.loc11_15.2 [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:     return %.loc11_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D(@Outer.%T.loc2: type, @Inner.%U.loc3: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%U) [symbolic = %Inner (constants.%Inner.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @D.%Inner (%Inner.4) = struct_value () [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @D.%Inner (%Inner.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_15.1: %.2 = struct_literal ()
// CHECK:STDOUT:     %.loc14_15.2: init @D.%Inner (%Inner.4) = class_init (), %return [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:     %.loc14_16: init @D.%Inner (%Inner.4) = converted %.loc14_15.1, %.loc14_15.2 [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:     return %.loc14_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@A.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer => constants.%Outer.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%U) {
// CHECK:STDOUT:   %T.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@B.%U) {
// CHECK:STDOUT:   %T.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Outer => constants.%Outer.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A.type => constants.%A.type.1
// CHECK:STDOUT:   %A => constants.%A.1
// CHECK:STDOUT:   %B.type => constants.%B.type.1
// CHECK:STDOUT:   %B => constants.%B.1
// CHECK:STDOUT:   %C.type => constants.%C.type.1
// CHECK:STDOUT:   %C => constants.%C.1
// CHECK:STDOUT:   %D.type => constants.%D.type.1
// CHECK:STDOUT:   %D => constants.%D.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@C.%T) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner => constants.%Inner.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %A.type => constants.%A.type.2
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT:   %B.type => constants.%B.type.2
// CHECK:STDOUT:   %B => constants.%B.2
// CHECK:STDOUT:   %C.type => constants.%C.type.2
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %D.type => constants.%D.type.2
// CHECK:STDOUT:   %D => constants.%D.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@D.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner => constants.%Inner.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@Inner.%T, @Inner.%U.1) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%U, constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:

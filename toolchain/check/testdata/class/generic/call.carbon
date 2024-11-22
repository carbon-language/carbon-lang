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

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);

// --- fail_too_few.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+7]]:8: error: 1 argument passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var a: Class(i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_few.carbon:[[@LINE-5]]:1: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*);

// --- fail_too_many.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+7]]:8: error: 3 arguments passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var a: Class(i32*, 1, 2);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_many.carbon:[[@LINE-5]]:1: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*, 1, 2);

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+9]]:8: error: cannot implicitly convert from `Core.IntLiteral` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+6]]:8: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N.1) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%.4, %.32) [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%empty_tuple.type, %.37) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: %i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc6_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_17.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_17.2: type = converted %int.make_type_signed, %.loc6_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_17.3: type = ptr_type %i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_20: Core.IntLiteral = int_value 5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc6_13.1: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_13.2: <bound method> = bound_method %.loc6_20, %.loc6_13.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_13.3: <specific function> = specific_function %.loc6_13.2, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %i32 = call %.loc6_13.3(%.loc6_20) [template = constants.%.32]
// CHECK:STDOUT:   %.loc6_13.4: %i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc6_13.5: %i32 = converted %.loc6_20, %.loc6_13.4 [template = constants.%.32]
// CHECK:STDOUT:   %Class.loc6: type = class_type @Class, @Class(constants.%.4, constants.%.32) [template = constants.%Class.3]
// CHECK:STDOUT:   %a.var: ref %Class.3 = var a
// CHECK:STDOUT:   %a: ref %Class.3 = bind_name a, %a.var
// CHECK:STDOUT:   %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc9_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_18: Core.IntLiteral = int_value 0 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_13.1: type = converted %.loc9_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc9_13.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_13.3: <bound method> = bound_method %.loc9_18, %.loc9_13.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_13.4: <specific function> = specific_function %.loc9_13.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %.loc9_13.4(%.loc9_18) [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.5: %i32 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.6: %i32 = converted %.loc9_18, %.loc9_13.5 [template = constants.%.37]
// CHECK:STDOUT:   %Class.loc9: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%.37) [template = constants.%Class.4]
// CHECK:STDOUT:   %b.var: ref %Class.4 = var b
// CHECK:STDOUT:   %b: ref %Class.4 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: %i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: %i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.1) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.4, constants.%.32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.4
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.4
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%.32
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%.32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%empty_tuple.type, constants.%.37) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%.37
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%.37
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: %i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc13_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc13_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_signed, %.loc13_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type %i32 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: %i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: %i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: %i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc13_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc13_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_signed, %.loc13_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type %i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_20: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_23: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: %i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: %i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N.1) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc4_23.1: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: %i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc15_14: Core.IntLiteral = int_value 5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_17: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc15_17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_20.2: type = converted %int.make_type_signed, %.loc15_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_20.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_13: type = converted %.loc15_14, <error> [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: %i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: %i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: %i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.1) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %Outer.1: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.1: %Inner.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %A.type.1: type = fn_type @A, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %A.1: %A.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Outer.3: type = class_type @Outer, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %B.type.1: type = fn_type @B, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %B.1: %B.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.3: type = class_type @Inner, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %C.type.1: type = fn_type @C, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %C.1: %C.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %D.type.1: type = fn_type @D, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %D.1: %D.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %struct.1: %Outer.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.type.2: type = generic_class_type @Inner, @Outer(%U) [symbolic]
// CHECK:STDOUT:   %Inner.4: %Inner.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.2: %Outer.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %A.type.2: type = fn_type @A, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %A.2: %A.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %B.type.2: type = fn_type @B, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %B.2: %B.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.type.2: type = fn_type @C, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %C.2: %C.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %D.type.2: type = fn_type @D, @Inner(%T, %T) [symbolic]
// CHECK:STDOUT:   %D.2: %D.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.3: %Inner.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct.4: %Inner.2 = struct_value () [symbolic]
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
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [template = constants.%Outer.1] {
// CHECK:STDOUT:     %T.patt.loc2_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc2_13.1, runtime_param<invalid> [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T.loc2_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc2_13.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner: @Outer.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner (constants.%Inner.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.1) = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.1)] {
// CHECK:STDOUT:       %U.patt.loc3_15.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_15.2 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc3_15.1, runtime_param<invalid> [symbolic = %U.patt.loc3_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc3_15.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc3_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc17: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc2_13.1: type, %U.loc3_15.1: type) {
// CHECK:STDOUT:   %U.loc3_15.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc3_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc3_15.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_15.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A.type: type = fn_type @A, @Inner(%T, %U.loc3_15.2) [symbolic = %A.type (constants.%A.type.1)]
// CHECK:STDOUT:   %A: @Inner.%A.type (%A.type.1) = struct_value () [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %B.type: type = fn_type @B, @Inner(%T, %U.loc3_15.2) [symbolic = %B.type (constants.%B.type.1)]
// CHECK:STDOUT:   %B: @Inner.%B.type (%B.type.1) = struct_value () [symbolic = %B (constants.%B.1)]
// CHECK:STDOUT:   %C.type: type = fn_type @C, @Inner(%T, %U.loc3_15.2) [symbolic = %C.type (constants.%C.type.1)]
// CHECK:STDOUT:   %C: @Inner.%C.type (%C.type.1) = struct_value () [symbolic = %C (constants.%C.1)]
// CHECK:STDOUT:   %D.type: type = fn_type @D, @Inner(%T, %U.loc3_15.2) [symbolic = %D.type (constants.%D.type.1)]
// CHECK:STDOUT:   %D: @Inner.%D.type (%D.type.1) = struct_value () [symbolic = %D (constants.%D.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %A.decl: @Inner.%A.type (%A.type.1) = fn_decl @A [symbolic = @Inner.%A (constants.%A.1)] {
// CHECK:STDOUT:       %return.patt: @A.%Outer.loc4_20.1 (%Outer.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @A.%Outer.loc4_20.1 (%Outer.2) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Outer.loc4_20.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc4_20.1 (constants.%Outer.2)]
// CHECK:STDOUT:       %return.param: ref @A.%Outer.loc4_20.1 (%Outer.2) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @A.%Outer.loc4_20.1 (%Outer.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.decl: @Inner.%B.type (%B.type.1) = fn_decl @B [symbolic = @Inner.%B (constants.%B.1)] {
// CHECK:STDOUT:       %return.patt: @B.%Outer.loc7_20.1 (%Outer.3) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @B.%Outer.loc7_20.1 (%Outer.3) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_15.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Outer.loc7_20.2: type = class_type @Outer, @Outer(constants.%U) [symbolic = %Outer.loc7_20.1 (constants.%Outer.3)]
// CHECK:STDOUT:       %return.param: ref @B.%Outer.loc7_20.1 (%Outer.3) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @B.%Outer.loc7_20.1 (%Outer.3) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %C.decl: @Inner.%C.type (%C.type.1) = fn_decl @C [symbolic = @Inner.%C (constants.%C.1)] {
// CHECK:STDOUT:       %return.patt: @C.%Inner.loc10_20.1 (%Inner.3) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @C.%Inner.loc10_20.1 (%Inner.3) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10: @C.%Inner.type (%Inner.type.1) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.loc10_15 (constants.%Inner.1)]
// CHECK:STDOUT:       %Inner.ref: @C.%Inner.type (%Inner.type.1) = name_ref Inner, %.loc10 [symbolic = %Inner.loc10_15 (constants.%Inner.1)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Inner.loc10_20.2: type = class_type @Inner, @Inner(constants.%T, constants.%T) [symbolic = %Inner.loc10_20.1 (constants.%Inner.3)]
// CHECK:STDOUT:       %return.param: ref @C.%Inner.loc10_20.1 (%Inner.3) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @C.%Inner.loc10_20.1 (%Inner.3) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.decl: @Inner.%D.type (%D.type.1) = fn_decl @D [symbolic = @Inner.%D (constants.%D.1)] {
// CHECK:STDOUT:       %return.patt: @D.%Inner.loc13_20.1 (%Inner.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @D.%Inner.loc13_20.1 (%Inner.2) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13: @D.%Inner.type (%Inner.type.1) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.loc13_15 (constants.%Inner.1)]
// CHECK:STDOUT:       %Inner.ref: @D.%Inner.type (%Inner.type.1) = name_ref Inner, %.loc13 [symbolic = %Inner.loc13_15 (constants.%Inner.1)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @Inner.%U.loc3_15.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_20.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_20.1 (constants.%Inner.2)]
// CHECK:STDOUT:       %return.param: ref @D.%Inner.loc13_20.1 (%Inner.2) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @D.%Inner.loc13_20.1 (%Inner.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc16: <witness> = complete_type_witness %.1 [template = constants.%.2]
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
// CHECK:STDOUT: generic fn @A(@Outer.%T.loc2_13.1: type, @Inner.%U.loc3_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer.loc4_20.1: type = class_type @Outer, @Outer(%T) [symbolic = %Outer.loc4_20.1 (constants.%Outer.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @A.%Outer.loc4_20.1 (%Outer.2) = struct_value () [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @A.%Outer.loc4_20.1 (%Outer.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc5_15.2: init @A.%Outer.loc4_20.1 (%Outer.2) = class_init (), %return [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:     %.loc5_16: init @A.%Outer.loc4_20.1 (%Outer.2) = converted %.loc5_15.1, %.loc5_15.2 [symbolic = %struct (constants.%struct.1)]
// CHECK:STDOUT:     return %.loc5_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B(@Outer.%T.loc2_13.1: type, @Inner.%U.loc3_15.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Outer.loc7_20.1: type = class_type @Outer, @Outer(%U) [symbolic = %Outer.loc7_20.1 (constants.%Outer.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @B.%Outer.loc7_20.1 (%Outer.3) = struct_value () [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @B.%Outer.loc7_20.1 (%Outer.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc8_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc8_15.2: init @B.%Outer.loc7_20.1 (%Outer.3) = class_init (), %return [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:     %.loc8_16: init @B.%Outer.loc7_20.1 (%Outer.3) = converted %.loc8_15.1, %.loc8_15.2 [symbolic = %struct (constants.%struct.2)]
// CHECK:STDOUT:     return %.loc8_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C(@Outer.%T.loc2_13.1: type, @Inner.%U.loc3_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner.loc10_15: @C.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner.loc10_15 (constants.%Inner.1)]
// CHECK:STDOUT:   %Inner.loc10_20.1: type = class_type @Inner, @Inner(%T, %T) [symbolic = %Inner.loc10_20.1 (constants.%Inner.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @C.%Inner.loc10_20.1 (%Inner.3) = struct_value () [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @C.%Inner.loc10_20.1 (%Inner.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_15.2: init @C.%Inner.loc10_20.1 (%Inner.3) = class_init (), %return [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:     %.loc11_16: init @C.%Inner.loc10_20.1 (%Inner.3) = converted %.loc11_15.1, %.loc11_15.2 [symbolic = %struct (constants.%struct.3)]
// CHECK:STDOUT:     return %.loc11_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D(@Outer.%T.loc2_13.1: type, @Inner.%U.loc3_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner.loc13_15: @D.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner.loc13_15 (constants.%Inner.1)]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Inner.loc13_20.1: type = class_type @Inner, @Inner(%T, %U) [symbolic = %Inner.loc13_20.1 (constants.%Inner.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @D.%Inner.loc13_20.1 (%Inner.2) = struct_value () [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @D.%Inner.loc13_20.1 (%Inner.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_15.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc14_15.2: init @D.%Inner.loc13_20.1 (%Inner.2) = class_init (), %return [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:     %.loc14_16: init @D.%Inner.loc13_20.1 (%Inner.2) = converted %.loc14_15.1, %.loc14_15.2 [symbolic = %struct (constants.%struct.4)]
// CHECK:STDOUT:     return %.loc14_16 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner => constants.%Inner.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%U
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
// CHECK:STDOUT: specific @Outer(@A.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer.loc4_20.1 => constants.%Outer.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%U) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner => constants.%Inner.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@B.%U) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Outer.loc7_20.1 => constants.%Outer.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%T) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%T
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
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
// CHECK:STDOUT: specific @Outer(@C.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@C.%T, @C.%T) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%T
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner.loc10_15 => constants.%Inner.1
// CHECK:STDOUT:   %Inner.loc10_20.1 => constants.%Inner.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@D.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@D.%T, @D.%U) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner.loc13_15 => constants.%Inner.1
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Inner.loc13_20.1 => constants.%Inner.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(%T, %U.loc3_15.2) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(%T.loc2_13.2) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

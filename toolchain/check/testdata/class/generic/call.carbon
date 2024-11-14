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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.1: i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N.1) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.9: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.10: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.10)> [symbolic]
// CHECK:STDOUT:   %N.patt.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.12: type = int_type unsigned, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.12)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.20: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.21: %.20 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.10)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.12)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.24: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.25: %.24 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 5 [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(%.3, %.30) [template]
// CHECK:STDOUT:   %.31: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.32, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, @Class(%empty_tuple.type, %.34) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.5) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.28]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.26 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.32 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
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
// CHECK:STDOUT:     %N.patt.loc4_23.1: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc6: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc6_17.2: type = converted %int.make_type_32, %.loc6_17.1 [template = i32]
// CHECK:STDOUT:   %.loc6_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_20: Core.IntLiteral = int_value 5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_13.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc6_13.2: <bound method> = bound_method %.loc6_20, %.loc6_13.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc6: init i32 = call %.loc6_13.2(%.loc6_20) [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_13.3: i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_13.4: i32 = converted %.loc6_20, %.loc6_13.3 [template = constants.%.30]
// CHECK:STDOUT:   %Class.loc6: type = class_type @Class, @Class(constants.%.3, constants.%.30) [template = constants.%Class.3]
// CHECK:STDOUT:   %a.var: ref %Class.3 = var a
// CHECK:STDOUT:   %a: ref %Class.3 = bind_name a, %a.var
// CHECK:STDOUT:   %Class.ref.loc9: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc9_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_18: Core.IntLiteral = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_13.1: type = converted %.loc9_15, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc9_13.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc9_13.3: <bound method> = bound_method %.loc9_18, %.loc9_13.2 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc9: init i32 = call %.loc9_13.3(%.loc9_18) [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_13.4: i32 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_13.5: i32 = converted %.loc9_18, %.loc9_13.4 [template = constants.%.34]
// CHECK:STDOUT:   %Class.loc9: type = class_type @Class, @Class(constants.%empty_tuple.type, constants.%.34) [template = constants.%Class.4]
// CHECK:STDOUT:   %b.var: ref %Class.4 = var b
// CHECK:STDOUT:   %b: ref %Class.4 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.18)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.18) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.10) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.10)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.4)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.10) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.10)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.1) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.10) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.9
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %.2 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.10) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.10
// CHECK:STDOUT:   %Dest.patt => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT:   %.2 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%.3, constants.%.30) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.3
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.3
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%.30
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%.30
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%empty_tuple.type, constants.%.34) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%.34
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%.34
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %N.patt.loc4_23.1: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %N.patt.loc4_23.1: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_20: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_23: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt.1: i32 = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T, %N.1) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(type)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(type) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.9: %.5 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.3 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.11 [template]
// CHECK:STDOUT:   %N.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.12: type = int_type signed, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.12)> [symbolic]
// CHECK:STDOUT:   %N.patt.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @impl.2(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %.14: type = int_type unsigned, %N.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%.14)> [symbolic]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.3, @impl.3(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: <witness> = interface_witness (%Convert.5) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type %ImplicitAs.type.7, %Convert.type.6 [template]
// CHECK:STDOUT:   %.17: %.16 = assoc_entity element0, imports.%import_ref.21 [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.4, @impl.5(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.5, @impl.6(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: <witness> = interface_witness (%Convert.8) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: type = assoc_entity_type %As.type.2, %Convert.type.9 [symbolic]
// CHECK:STDOUT:   %.21: %.20 = assoc_entity element0, imports.%import_ref.34 [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %.22: type = assoc_entity_type %As.type.3, %Convert.type.10 [template]
// CHECK:STDOUT:   %.23: %.22 = assoc_entity element0, imports.%import_ref.35 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.12)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.7, @impl.8(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.14)> [symbolic]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.8, @impl.9(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [symbolic]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.12) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.26: type = assoc_entity_type %As.type.6, %Convert.type.13 [template]
// CHECK:STDOUT:   %.27: %.26 = assoc_entity element0, imports.%import_ref.45 [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.9, @impl.11(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.10, @impl.12(%N.2) [symbolic]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [symbolic]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.5) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//prelude/operators/as, inst+133, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.14 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.16: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.6)]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.19: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.23: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.24 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.7]
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.30 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.32: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.37: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.38 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.40: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.41 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.43: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.45 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.47: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.48 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.50: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.51 = import_ref Core//prelude/operators/as, inst+648, unloaded
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
// CHECK:STDOUT:     %N.patt.loc4_23.1: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: i32 = value_param_pattern %N.patt.loc4_23.1, runtime_param<invalid> [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %N.param: i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_23.1: i32 = bind_symbolic_name N, 1, %N.param [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %.loc15_14: Core.IntLiteral = int_value 5 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc15_20.2: type = converted %int.make_type_32, %.loc15_20.1 [template = i32]
// CHECK:STDOUT:   %.loc15_20.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_13: type = converted %.loc15_14, <error> [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.5) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.9)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.9) = struct_value () [symbolic = %Convert (constants.%Convert.9)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.9) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.20) = assoc_entity element0, imports.%import_ref.34 [symbolic = %.2 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.28
// CHECK:STDOUT:     .Convert = imports.%import_ref.29
// CHECK:STDOUT:     witness = (imports.%import_ref.30)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.12 as imports.%import_ref.13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.5)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.5) = struct_value () [symbolic = %Convert (constants.%Convert.5)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.15 as imports.%import_ref.16 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.17
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.18 as imports.%import_ref.19 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.22 as imports.%import_ref.23 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.31 as imports.%import_ref.32 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.36 as imports.%import_ref.37 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.12)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.12) = struct_value () [symbolic = %Convert (constants.%Convert.12)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.39 as imports.%import_ref.40 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.41
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.42 as imports.%import_ref.43 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.46 as imports.%import_ref.47 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.48
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.15)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.15) = struct_value () [symbolic = %Convert (constants.%Convert.15)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.29)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.49 as imports.%import_ref.50 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.51
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type, %N.loc4_23.1: i32) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc4_23.2: i32 = bind_symbolic_name N, 1 [symbolic = %N.loc4_23.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_23.2: i32 = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc4_23.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_33: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.4)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.12) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.12)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.2)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T, constants.%N.1) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %N.loc4_23.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_23.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(type) {
// CHECK:STDOUT:   %Dest => type
// CHECK:STDOUT:   %Dest.patt => type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6
// CHECK:STDOUT:   %Convert => constants.%Convert.6
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.10
// CHECK:STDOUT:   %Convert => constants.%Convert.10
// CHECK:STDOUT:   %.1 => constants.%.22
// CHECK:STDOUT:   %.2 => constants.%.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.12) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.12
// CHECK:STDOUT:   %Dest.patt => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.13
// CHECK:STDOUT:   %.1 => constants.%.26
// CHECK:STDOUT:   %.2 => constants.%.27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %N.patt => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N.2) {
// CHECK:STDOUT:   %N => constants.%N.2
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_in_nested_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
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
// CHECK:STDOUT: specific @Inner(@Inner.%T, @Inner.%U.loc3_15.2) {
// CHECK:STDOUT:   %U.loc3_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc3_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@Outer.%T.loc2_13.2) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

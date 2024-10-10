// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_abstract.carbon


// --- fail_rejects_valid_abstract_subobject_construction.carbon
abstract class Abstract {
  var a: i32;
}

class Derived {
  extend base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_rejects_valid_abstract_subobject_construction.carbon:[[@LINE+8]]:19: error: cannot construct instance of abstract class; consider using `partial Abstract` instead
  // CHECK:STDERR:   return {.base = {.a = 1}, .d = 7};
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: base_abstract_subobject_access.carbon: error: `Main//default` previously provided by `fail_rejects_valid_abstract_subobject_construction.carbon`
  // CHECK:STDERR:
  // CHECK:STDERR: fail_abstract_decl.carbon: error: `Main//default` previously provided by `fail_rejects_valid_abstract_subobject_construction.carbon`
  // CHECK:STDERR:
  return {.base = {.a = 1}, .d = 7};
}

// --- base_abstract_subobject_access.carbon
abstract class Abstract {
  var a: i32;
}

class Derived {
  extend base: Abstract;

  var d: i32;
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.a);
}

// --- fail_abstract_decl.carbon
// CHECK:STDERR: fail_abstract_decl.carbon:[[@LINE+3]]:1: error: `abstract` not allowed on `class` declaration, only definition
// CHECK:STDERR: abstract class AbstractDecl;
// CHECK:STDERR: ^~~~~~~~
abstract class AbstractDecl;

// CHECK:STDOUT: --- fail_rejects_valid_abstract_subobject_construction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Abstract, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Abstract [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Abstract, .d: i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5, .d: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: %.3, .d: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
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
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {} {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %return: ref %Derived = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc2_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc2_10.2: type = converted %int.make_type_32, %.loc2_10.1 [template = i32]
// CHECK:STDOUT:   %.loc2_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc3: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .a = %.loc2_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc6: %.6 = base_decl %Abstract, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc6
// CHECK:STDOUT:   .d = %.loc8_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_25: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc21_26: %.3 = struct_literal (%.loc21_25)
// CHECK:STDOUT:   %.loc21_34: i32 = int_literal 7 [template = constants.%.14]
// CHECK:STDOUT:   %.loc21_35: %.15 = struct_literal (%.loc21_26, %.loc21_34)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_abstract_subobject_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Abstract, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %Derived, %Abstract [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Abstract, .d: i32} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %.10: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: %.5, .d: i32} [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.15: type = ptr_type %.8 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
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
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.param: %Derived = param d, runtime_param0
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_35.1: %.10 = tuple_literal (%int.make_type_32.loc11_27, %int.make_type_32.loc11_32)
// CHECK:STDOUT:     %.loc11_35.2: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_35.3: type = converted %int.make_type_32.loc11_27, %.loc11_35.2 [template = i32]
// CHECK:STDOUT:     %.loc11_35.4: type = value_of_initializer %int.make_type_32.loc11_32 [template = i32]
// CHECK:STDOUT:     %.loc11_35.5: type = converted %int.make_type_32.loc11_32, %.loc11_35.4 [template = i32]
// CHECK:STDOUT:     %.loc11_35.6: type = converted %.loc11_35.1, constants.%.11 [template = constants.%.11]
// CHECK:STDOUT:     %return: ref %.11 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc2_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc2_10.2: type = converted %int.make_type_32, %.loc2_10.1 [template = i32]
// CHECK:STDOUT:   %.loc2_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc3: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .a = %.loc2_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc6: %.6 = base_decl %Abstract, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc6
// CHECK:STDOUT:   .d = %.loc8_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: %Derived) -> %return: %.11 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc12_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc12_12: %.7 = name_ref d, @Derived.%.loc8_8 [template = @Derived.%.loc8_8]
// CHECK:STDOUT:   %.loc12_12.1: ref i32 = class_element_access %d.ref.loc12_11, element1
// CHECK:STDOUT:   %.loc12_12.2: i32 = bind_value %.loc12_12.1
// CHECK:STDOUT:   %d.ref.loc12_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %.6 = name_ref base, @Derived.%.loc6 [template = @Derived.%.loc6]
// CHECK:STDOUT:   %.loc12_17.1: ref %Abstract = class_element_access %d.ref.loc12_16, element0
// CHECK:STDOUT:   %.loc12_17.2: %Abstract = bind_value %.loc12_17.1
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Abstract.%.loc2_8 [template = @Abstract.%.loc2_8]
// CHECK:STDOUT:   %.loc12_22.1: ref i32 = class_element_access %.loc12_17.2, element0
// CHECK:STDOUT:   %.loc12_22.2: i32 = bind_value %.loc12_22.1
// CHECK:STDOUT:   %.loc12_24.1: %.11 = tuple_literal (%.loc12_12.2, %.loc12_22.2)
// CHECK:STDOUT:   %.loc12_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc12_24.3: init i32 = initialize_from %.loc12_12.2 to %.loc12_24.2
// CHECK:STDOUT:   %.loc12_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc12_24.5: init i32 = initialize_from %.loc12_22.2 to %.loc12_24.4
// CHECK:STDOUT:   %.loc12_24.6: init %.11 = tuple_init (%.loc12_24.3, %.loc12_24.5) to %return
// CHECK:STDOUT:   %.loc12_25: init %.11 = converted %.loc12_24.1, %.loc12_24.6
// CHECK:STDOUT:   return %.loc12_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractDecl: type = class_type @AbstractDecl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AbstractDecl = %AbstractDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AbstractDecl.decl: type = class_decl @AbstractDecl [template = constants.%AbstractDecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractDecl;
// CHECK:STDOUT:

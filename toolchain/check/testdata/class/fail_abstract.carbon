// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_abstract.carbon

abstract class Abstract {
  var a: i32;
}

class Derived {
  extend base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+3]]:19: error: cannot construct instance of abstract class; consider using `partial Abstract` instead
  // CHECK:STDERR:   return {.base = {.a = 1}, .d = 7};
  // CHECK:STDERR:                   ^~~~~~~~
  return {.base = {.a = 1}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.a);
}

// CHECK:STDOUT: --- fail_abstract.carbon
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
// CHECK:STDOUT:   %.16: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.17: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT:   %.18: type = ptr_type %.17 [template]
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
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %Derived.ref.loc21: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     @Make.%return: ref %Derived = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %Derived.ref.loc29: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc29_11.1: %Derived = param d, runtime_param0
// CHECK:STDOUT:     @Access.%d: %Derived = bind_name d, %d.loc29_11.1
// CHECK:STDOUT:     %int.make_type_32.loc29_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc29_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_35.1: %.16 = tuple_literal (%int.make_type_32.loc29_27, %int.make_type_32.loc29_32)
// CHECK:STDOUT:     %.loc29_35.2: type = value_of_initializer %int.make_type_32.loc29_27 [template = i32]
// CHECK:STDOUT:     %.loc29_35.3: type = converted %int.make_type_32.loc29_27, %.loc29_35.2 [template = i32]
// CHECK:STDOUT:     %.loc29_35.4: type = value_of_initializer %int.make_type_32.loc29_32 [template = i32]
// CHECK:STDOUT:     %.loc29_35.5: type = converted %int.make_type_32.loc29_32, %.loc29_35.4 [template = i32]
// CHECK:STDOUT:     %.loc29_35.6: type = converted %.loc29_35.1, constants.%.17 [template = constants.%.17]
// CHECK:STDOUT:     @Access.%return: ref %.17 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc16: %.6 = base_decl %Abstract, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc18_10.2: type = converted %int.make_type_32, %.loc18_10.1 [template = i32]
// CHECK:STDOUT:   %.loc18_8: %.7 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .d = %.loc18_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_25: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc26_26: %.3 = struct_literal (%.loc26_25)
// CHECK:STDOUT:   %.loc26_34: i32 = int_literal 7 [template = constants.%.14]
// CHECK:STDOUT:   %.loc26_35: %.15 = struct_literal (%.loc26_26, %.loc26_34)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: %Derived) -> %return: %.17 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc30_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc30_12: %.7 = name_ref d, @Derived.%.loc18_8 [template = @Derived.%.loc18_8]
// CHECK:STDOUT:   %.loc30_12.1: ref i32 = class_element_access %d.ref.loc30_11, element1
// CHECK:STDOUT:   %.loc30_12.2: i32 = bind_value %.loc30_12.1
// CHECK:STDOUT:   %d.ref.loc30_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %.6 = name_ref base, @Derived.%.loc16 [template = @Derived.%.loc16]
// CHECK:STDOUT:   %.loc30_17.1: ref %Abstract = class_element_access %d.ref.loc30_16, element0
// CHECK:STDOUT:   %.loc30_17.2: %Abstract = bind_value %.loc30_17.1
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Abstract.%.loc12_8 [template = @Abstract.%.loc12_8]
// CHECK:STDOUT:   %.loc30_22.1: ref i32 = class_element_access %.loc30_17.2, element0
// CHECK:STDOUT:   %.loc30_22.2: i32 = bind_value %.loc30_22.1
// CHECK:STDOUT:   %.loc30_24.1: %.17 = tuple_literal (%.loc30_12.2, %.loc30_22.2)
// CHECK:STDOUT:   %.loc30_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc30_24.3: init i32 = initialize_from %.loc30_12.2 to %.loc30_24.2
// CHECK:STDOUT:   %.loc30_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc30_24.5: init i32 = initialize_from %.loc30_22.2 to %.loc30_24.4
// CHECK:STDOUT:   %.loc30_24.6: init %.17 = tuple_init (%.loc30_24.3, %.loc30_24.5) to %return
// CHECK:STDOUT:   %.loc30_25: init %.17 = converted %.loc30_24.1, %.loc30_24.6
// CHECK:STDOUT:   return %.loc30_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

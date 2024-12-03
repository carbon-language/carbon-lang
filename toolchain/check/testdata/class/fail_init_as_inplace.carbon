// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_init_as_inplace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_init_as_inplace.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn G(p: Class*);

fn F() {
  // TODO: This case should presumably work: `{...} as Class` should be an
  // initializing expression, not a value expression.
  //
  // CHECK:STDERR: fail_init_as_inplace.carbon:[[@LINE+3]]:18: error: cannot copy value of type `Class` [CopyOfUncopyableType]
  // CHECK:STDERR:   var c: Class = {.a = 1, .b = 2} as Class;
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~
  var c: Class = {.a = 1, .b = 2} as Class;
  G(&c);
}

// CHECK:STDOUT: --- fail_init_as_inplace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %Class [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %Class = struct_value (%int.1.2, %int.2.2) [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type %Class [template = constants.%.4]
// CHECK:STDOUT:     %p.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.4 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.1 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int.32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %int.make_type_signed.loc13, %.loc13_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8: %.1 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT:   complete_type_witness = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.4);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref.loc25_10: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref %Class = var c
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc25_33.1: %.6 = struct_literal (%int.1, %int.2)
// CHECK:STDOUT:   %Class.ref.loc25_38: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc25_33.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc25_33.3: <bound method> = bound_method %int.1, %.loc25_33.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc25_33.4: <specific function> = specific_function %.loc25_33.3, @Convert.2(constants.%int.32) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc25_33.1: init %i32 = call %.loc25_33.4(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc25_33.5: init %i32 = converted %int.1, %int.convert_checked.loc25_33.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc25_33.6: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc25_33.7: ref %i32 = class_element_access %.loc25_33.6, element0
// CHECK:STDOUT:   %.loc25_33.8: init %i32 = initialize_from %.loc25_33.5 to %.loc25_33.7 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc25_33.9: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc25_33.10: <bound method> = bound_method %int.2, %.loc25_33.9 [template = constants.%.33]
// CHECK:STDOUT:   %.loc25_33.11: <specific function> = specific_function %.loc25_33.10, @Convert.2(constants.%int.32) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc25_33.2: init %i32 = call %.loc25_33.11(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc25_33.12: init %i32 = converted %int.2, %int.convert_checked.loc25_33.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc25_33.13: ref %i32 = class_element_access %.loc25_33.6, element1
// CHECK:STDOUT:   %.loc25_33.14: init %i32 = initialize_from %.loc25_33.12 to %.loc25_33.13 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc25_33.15: init %Class = class_init (%.loc25_33.8, %.loc25_33.14), %.loc25_33.6 [template = constants.%struct]
// CHECK:STDOUT:   %.loc25_33.16: ref %Class = temporary %.loc25_33.6, %.loc25_33.15
// CHECK:STDOUT:   %.loc25_35.1: ref %Class = converted %.loc25_33.1, %.loc25_33.16
// CHECK:STDOUT:   %.loc25_35.2: %Class = bind_value %.loc25_35.1
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %.loc26: %.4 = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref(%.loc26)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

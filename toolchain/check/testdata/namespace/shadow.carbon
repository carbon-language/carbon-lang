// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/shadow.carbon

fn A();

namespace N;
fn N.A();

namespace N.M;

fn N.M.B() -> i32 {
  // This is N.A, not package.A.
  A();
  if (true) {
    var A: i32 = 0;
    // This is the local A.
    // TODO: Decide if we should warn or error on the shadowing of N.A here.
    return A;
  }
  return 0;
}

// CHECK:STDOUT: --- shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.1: type = fn_type @A.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type.1 = struct_value () [template]
// CHECK:STDOUT:   %A.type.2: type = fn_type @A.2 [template]
// CHECK:STDOUT:   %A.2: %A.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: %A.type.1 = fn_decl @A.1 [template = constants.%A.1] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl.loc14
// CHECK:STDOUT:     .M = %M
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc14: %A.type.2 = fn_decl @A.2 [template = constants.%A.2] {} {}
// CHECK:STDOUT:   %M: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_15.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_15.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_15.3: type = converted %int.make_type_signed.loc18, %.loc18_15.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc20: %A.type.2 = name_ref A, file.%A.decl.loc14 [template = constants.%A.2]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref.loc20()
// CHECK:STDOUT:   %.loc21: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc21 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc22_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_12.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_12.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_12.3: type = converted %int.make_type_signed.loc22, %.loc22_12.2 [template = constants.%i32]
// CHECK:STDOUT:   %A.var: ref %i32 = var A
// CHECK:STDOUT:   %A: ref %i32 = bind_name A, %A.var
// CHECK:STDOUT:   %.loc22_18: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_19.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_19.2: <bound method> = bound_method %.loc22_18, %.loc22_19.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc22_19.3: <specific function> = specific_function %.loc22_19.2, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %.loc22_19.3(%.loc22_18) [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_19.4: init %i32 = converted %.loc22_18, %int.convert_checked.loc22 [template = constants.%.30]
// CHECK:STDOUT:   assign %A.var, %.loc22_19.4
// CHECK:STDOUT:   %A.ref.loc25: ref %i32 = name_ref A, %A
// CHECK:STDOUT:   %.loc25: %i32 = bind_value %A.ref.loc25
// CHECK:STDOUT:   return %.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc27_10: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc27_11.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc27_11.2: <bound method> = bound_method %.loc27_10, %.loc27_11.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc27_11.3: <specific function> = specific_function %.loc27_11.2, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %.loc27_11.3(%.loc27_10) [template = constants.%.30]
// CHECK:STDOUT:   %.loc27_11.4: %i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%.30]
// CHECK:STDOUT:   %.loc27_11.5: %i32 = converted %.loc27_10, %.loc27_11.4 [template = constants.%.30]
// CHECK:STDOUT:   return %.loc27_11.5
// CHECK:STDOUT: }
// CHECK:STDOUT:

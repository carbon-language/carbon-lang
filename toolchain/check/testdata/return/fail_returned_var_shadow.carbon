// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn SameScope() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+7]]:18: ERROR: Cannot declare a `returned var` in the scope of another `returned var`.
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-4]]:18: `returned var` was declared here.
    // CHECK:STDERR:     returned var v: i32 = 0;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR:
    returned var w: i32 = 1;
  }
  return 0;
}

fn DifferentScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    if (true) {
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+6]]:20: ERROR: Cannot declare a `returned var` in the scope of another `returned var`.
      // CHECK:STDERR:       returned var w: i32 = 1;
      // CHECK:STDERR:                    ^
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-5]]:18: `returned var` was declared here.
      // CHECK:STDERR:     returned var v: i32 = 0;
      // CHECK:STDERR:                  ^
      returned var w: i32 = 1;
    }
  }
  return 0;
}

// CHECK:STDOUT: --- fail_returned_var_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %SameScope: type = fn_type @SameScope [template]
// CHECK:STDOUT:   %struct.2: SameScope = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %DifferentScopes: type = fn_type @DifferentScopes [template]
// CHECK:STDOUT:   %struct.3: DifferentScopes = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SameScope = %SameScope.decl
// CHECK:STDOUT:     .DifferentScopes = %DifferentScopes.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %SameScope.decl: SameScope = fn_decl @SameScope [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     @SameScope.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %DifferentScopes.decl: DifferentScopes = fn_decl @DifferentScopes [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc22_25.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:     %.loc22_25.2: type = converted %int.make_type_32.loc22, %.loc22_25.1 [template = i32]
// CHECK:STDOUT:     @DifferentScopes.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc8 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_32.loc9, %.loc9_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc9_27
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc17_21.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_21.2: type = converted %int.make_type_32.loc17, %.loc17_21.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc17_27: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc17_27
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc19: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc23 br !if.then.loc23 else br !if.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc23:
// CHECK:STDOUT:   %int.make_type_32.loc24: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc24_21.1: type = value_of_initializer %int.make_type_32.loc24 [template = i32]
// CHECK:STDOUT:   %.loc24_21.2: type = converted %int.make_type_32.loc24, %.loc24_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc24_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc24_27
// CHECK:STDOUT:   %.loc25: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc25 br !if.then.loc25 else br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc25:
// CHECK:STDOUT:   %int.make_type_32.loc32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc32_23.1: type = value_of_initializer %int.make_type_32.loc32 [template = i32]
// CHECK:STDOUT:   %.loc32_23.2: type = converted %int.make_type_32.loc32, %.loc32_23.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc32_29: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc32_29
// CHECK:STDOUT:   br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc25:
// CHECK:STDOUT:   br !if.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc23:
// CHECK:STDOUT:   %.loc35: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:

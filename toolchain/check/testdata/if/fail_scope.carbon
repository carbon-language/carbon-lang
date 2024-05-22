// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn VarScope(b: bool) -> i32 {
  if (b) {
    var n: i32 = 2;
    return n;
  }
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: ERROR: Name `n` not found.
  // CHECK:STDERR:   return n;
  // CHECK:STDERR:          ^
  return n;
}

// CHECK:STDOUT: --- fail_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Bool = struct_value () [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %VarScope: type = fn_type @VarScope [template]
// CHECK:STDOUT:   %struct.3: VarScope = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .VarScope = %VarScope.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Bool = import_ref ir2, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %VarScope.decl: VarScope = fn_decl @VarScope [template = constants.%struct.3] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc7_16.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc7_16.2: type = converted %bool.make_type, %.loc7_16.1 [template = bool]
// CHECK:STDOUT:     %b.loc7_13.1: bool = param b
// CHECK:STDOUT:     @VarScope.%b: bool = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:     %.loc7_25.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc7_25.2: type = converted %int.make_type_32, %.loc7_25.1 [template = i32]
// CHECK:STDOUT:     @VarScope.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @VarScope(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc9_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc9_12.2: type = converted %int.make_type_32, %.loc9_12.1 [template = i32]
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   assign %n.var, %.loc9_18
// CHECK:STDOUT:   %n.ref.loc10: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc10: i32 = bind_value %n.ref.loc10
// CHECK:STDOUT:   return %.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %n.ref.loc15: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

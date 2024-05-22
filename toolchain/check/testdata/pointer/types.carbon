// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT:   %Ptr: type = fn_type @Ptr [template]
// CHECK:STDOUT:   %struct.2: Ptr = struct_value () [template]
// CHECK:STDOUT:   %.3: type = const_type i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %ConstPtr: type = fn_type @ConstPtr [template]
// CHECK:STDOUT:   %struct.3: ConstPtr = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .ConstPtr = %ConstPtr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Ptr.decl: Ptr = fn_decl @Ptr [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_14.1: type = value_of_initializer %int.make_type_32.loc7_11 [template = i32]
// CHECK:STDOUT:     %.loc7_14.2: type = converted %int.make_type_32.loc7_11, %.loc7_14.1 [template = i32]
// CHECK:STDOUT:     %.loc7_14.3: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %p.loc7_8.1: i32* = param p
// CHECK:STDOUT:     @Ptr.%p: i32* = bind_name p, %p.loc7_8.1
// CHECK:STDOUT:     %int.make_type_32.loc7_20: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_23.1: type = value_of_initializer %int.make_type_32.loc7_20 [template = i32]
// CHECK:STDOUT:     %.loc7_23.2: type = converted %int.make_type_32.loc7_20, %.loc7_23.1 [template = i32]
// CHECK:STDOUT:     %.loc7_23.3: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:     @Ptr.%return: ref i32* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %ConstPtr.decl: ConstPtr = fn_decl @ConstPtr [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_16.1: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_16.2: type = converted %int.make_type_32.loc11_22, %.loc11_16.1 [template = i32]
// CHECK:STDOUT:     %.loc11_16.3: type = const_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_25: type = ptr_type const i32 [template = constants.%.4]
// CHECK:STDOUT:     %p.loc11_13.1: const i32* = param p
// CHECK:STDOUT:     @ConstPtr.%p: const i32* = bind_name p, %p.loc11_13.1
// CHECK:STDOUT:     %int.make_type_32.loc11_38: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_32.1: type = value_of_initializer %int.make_type_32.loc11_38 [template = i32]
// CHECK:STDOUT:     %.loc11_32.2: type = converted %int.make_type_32.loc11_38, %.loc11_32.1 [template = i32]
// CHECK:STDOUT:     %.loc11_32.3: type = const_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_42: type = ptr_type const i32 [template = constants.%.4]
// CHECK:STDOUT:     @ConstPtr.%return: ref const i32* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p: i32*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p: const i32*) -> const i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32* = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

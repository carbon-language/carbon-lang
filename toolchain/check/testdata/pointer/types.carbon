// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/types.carbon

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [template]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = const_type i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [template]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [template]
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
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .ConstPtr = %ConstPtr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: %Ptr.type = fn_decl @Ptr [template = constants.%Ptr] {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_32.loc11_11, %.loc11_14.1 [template = i32]
// CHECK:STDOUT:     %.loc11_14.3: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %p.loc11_8.1: %.2 = param p
// CHECK:STDOUT:     @Ptr.%p: %.2 = bind_name p, %p.loc11_8.1
// CHECK:STDOUT:     %int.make_type_32.loc11_20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_23.1: type = value_of_initializer %int.make_type_32.loc11_20 [template = i32]
// CHECK:STDOUT:     %.loc11_23.2: type = converted %int.make_type_32.loc11_20, %.loc11_23.1 [template = i32]
// CHECK:STDOUT:     %.loc11_23.3: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:     @Ptr.%return: ref %.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [template = constants.%ConstPtr] {
// CHECK:STDOUT:     %int.make_type_32.loc15_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_16.1: type = value_of_initializer %int.make_type_32.loc15_22 [template = i32]
// CHECK:STDOUT:     %.loc15_16.2: type = converted %int.make_type_32.loc15_22, %.loc15_16.1 [template = i32]
// CHECK:STDOUT:     %.loc15_16.3: type = const_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_25: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %p.loc15_13.1: %.4 = param p
// CHECK:STDOUT:     @ConstPtr.%p: %.4 = bind_name p, %p.loc15_13.1
// CHECK:STDOUT:     %int.make_type_32.loc15_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %int.make_type_32.loc15_38 [template = i32]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %int.make_type_32.loc15_38, %.loc15_32.1 [template = i32]
// CHECK:STDOUT:     %.loc15_32.3: type = const_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_42: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     @ConstPtr.%return: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p: %.2) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.2 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p: %.4) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

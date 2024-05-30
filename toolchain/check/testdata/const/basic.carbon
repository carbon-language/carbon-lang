// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/const/basic.carbon

fn A(p: const i32**) -> const i32** {
  return p;
}

fn B(p: const (i32*)) -> const (i32*) {
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type const i32* [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.2: A = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: type = const_type i32* [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.3: B = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11_15 [template = i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_32.loc11_15, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:     %.loc11_9.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_18: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_19: type = ptr_type const i32* [template = constants.%.4]
// CHECK:STDOUT:     %p.loc11_6.1: const i32** = param p
// CHECK:STDOUT:     @A.%p: const i32** = bind_name p, %p.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_31: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_32.loc11_31 [template = i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_32.loc11_31, %.loc11_25.1 [template = i32]
// CHECK:STDOUT:     %.loc11_25.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_35: type = ptr_type const i32* [template = constants.%.4]
// CHECK:STDOUT:     @A.%return: ref const i32** = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc15_16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc15_19.1: type = value_of_initializer %int.make_type_32.loc15_16 [template = i32]
// CHECK:STDOUT:     %.loc15_19.2: type = converted %int.make_type_32.loc15_16, %.loc15_19.1 [template = i32]
// CHECK:STDOUT:     %.loc15_19.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_9: type = const_type i32* [template = constants.%.6]
// CHECK:STDOUT:     %p.loc15_6.1: const (i32*) = param p
// CHECK:STDOUT:     @B.%p: const (i32*) = bind_name p, %p.loc15_6.1
// CHECK:STDOUT:     %int.make_type_32.loc15_33: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_32.loc15_33 [template = i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_32.loc15_33, %.loc15_36.1 [template = i32]
// CHECK:STDOUT:     %.loc15_36.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_26: type = const_type i32* [template = constants.%.6]
// CHECK:STDOUT:     @B.%return: ref const (i32*) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p: const i32**) -> const i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32** = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p: const (i32*)) -> const (i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const (i32*) = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

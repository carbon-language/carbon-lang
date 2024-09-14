// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/basic.carbon

fn A(p: const i32**) -> const i32** {
  return p;
}

fn B(p: const (i32*)) -> const (i32*) {
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: type = const_type %.5 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %int.make_type_32.loc11_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11_15 [template = i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_32.loc11_15, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:     %.loc11_9.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_18: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_19: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %p.loc11_6.1: %.4 = param p, runtime_param0
// CHECK:STDOUT:     @A.%p: %.4 = bind_name p, %p.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_32.loc11_31 [template = i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_32.loc11_31, %.loc11_25.1 [template = i32]
// CHECK:STDOUT:     %.loc11_25.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_35: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     @A.%return: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc15_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_19.1: type = value_of_initializer %int.make_type_32.loc15_16 [template = i32]
// CHECK:STDOUT:     %.loc15_19.2: type = converted %int.make_type_32.loc15_16, %.loc15_19.1 [template = i32]
// CHECK:STDOUT:     %.loc15_19.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_9: type = const_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %p.loc15_6.1: %.6 = param p, runtime_param0
// CHECK:STDOUT:     @B.%p: %.6 = bind_name p, %p.loc15_6.1
// CHECK:STDOUT:     %int.make_type_32.loc15_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_32.loc15_33 [template = i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_32.loc15_33, %.loc15_36.1 [template = i32]
// CHECK:STDOUT:     %.loc15_36.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_26: type = const_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     @B.%return: ref %.6 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p: %.4) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p: %.6) -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

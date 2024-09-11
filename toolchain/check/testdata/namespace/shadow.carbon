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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type.1 = struct_value () [template]
// CHECK:STDOUT:   %A.type.2: type = fn_type @A.2 [template]
// CHECK:STDOUT:   %A.2: %A.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: %A.type.1 = fn_decl @A.1 [template = constants.%A.1] {}
// CHECK:STDOUT:   %N: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl.loc14
// CHECK:STDOUT:     .M = %M
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc14: %A.type.2 = fn_decl @A.2 [template = constants.%A.2] {}
// CHECK:STDOUT:   %M: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc18_15.2: type = converted %int.make_type_32, %.loc18_15.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc20: %A.type.2 = name_ref A, file.%A.decl.loc14 [template = constants.%A.2]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref.loc20()
// CHECK:STDOUT:   %.loc21: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc21 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc22_12.2: type = converted %int.make_type_32, %.loc22_12.1 [template = i32]
// CHECK:STDOUT:   %A.var: ref i32 = var A
// CHECK:STDOUT:   %A: ref i32 = bind_name A, %A.var
// CHECK:STDOUT:   %.loc22_18: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %A.var, %.loc22_18
// CHECK:STDOUT:   %A.ref.loc25: ref i32 = name_ref A, %A
// CHECK:STDOUT:   %.loc25: i32 = bind_value %A.ref.loc25
// CHECK:STDOUT:   return %.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:

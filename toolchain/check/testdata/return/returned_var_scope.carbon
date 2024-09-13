// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/returned_var_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/returned_var_scope.carbon

fn UnrelatedScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  if (true) {
    returned var w: i32 = 1;
  }
  return 0;
}

fn EnclosingButAfter(b: bool) -> i32 {
  if (b) {
    returned var v: i32 = 0;
    return var;
  }
  returned var w: i32 = 1;
  return var;
}

// CHECK:STDOUT: --- returned_var_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %UnrelatedScopes.type: type = fn_type @UnrelatedScopes [template]
// CHECK:STDOUT:   %UnrelatedScopes: %UnrelatedScopes.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %EnclosingButAfter.type: type = fn_type @EnclosingButAfter [template]
// CHECK:STDOUT:   %EnclosingButAfter: %EnclosingButAfter.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UnrelatedScopes = %UnrelatedScopes.decl
// CHECK:STDOUT:     .EnclosingButAfter = %EnclosingButAfter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %UnrelatedScopes.decl: %UnrelatedScopes.type = fn_decl @UnrelatedScopes [template = constants.%UnrelatedScopes] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_32.loc11, %.loc11_25.1 [template = i32]
// CHECK:STDOUT:     @UnrelatedScopes.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EnclosingButAfter.decl: %EnclosingButAfter.type = fn_decl @EnclosingButAfter [template = constants.%EnclosingButAfter] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc21_25.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc21_25.2: type = converted %bool.make_type, %.loc21_25.1 [template = bool]
// CHECK:STDOUT:     %b.loc21_22.1: bool = param b, runtime_param0
// CHECK:STDOUT:     @EnclosingButAfter.%b: bool = bind_name b, %b.loc21_22.1
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_34.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_34.2: type = converted %int.make_type_32.loc21, %.loc21_34.1 [template = i32]
// CHECK:STDOUT:     @EnclosingButAfter.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_32.loc13, %.loc13_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc13_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc13_27
// CHECK:STDOUT:   br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %.loc15: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc15 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_21.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_21.2: type = converted %int.make_type_32.loc16, %.loc16_21.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc16_27: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc16_27
// CHECK:STDOUT:   br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_21.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_21.2: type = converted %int.make_type_32.loc23, %.loc23_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc23_27
// CHECK:STDOUT:   %.loc23_18: i32 = bind_value %v
// CHECK:STDOUT:   return %.loc23_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_19.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:   %.loc26_19.2: type = converted %int.make_type_32.loc26, %.loc26_19.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc26_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc26_25
// CHECK:STDOUT:   %.loc26_16: i32 = bind_value %w
// CHECK:STDOUT:   return %.loc26_16
// CHECK:STDOUT: }
// CHECK:STDOUT:

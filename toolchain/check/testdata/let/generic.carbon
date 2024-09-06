// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/generic.carbon

fn F() {
  let T:! type = i32;
  var p: T*;
  var a: T = *p;
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_32, %.loc12_21.1 [template = i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0, %.loc12_21.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref.loc13: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc13: type = ptr_type %T [symbolic = constants.%.2]
// CHECK:STDOUT:   %p.var: ref %.2 = var p
// CHECK:STDOUT:   %p: ref %.2 = bind_name p, %p.var
// CHECK:STDOUT:   %T.ref.loc14: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %a.var: ref %T = var a
// CHECK:STDOUT:   %a: ref %T = bind_name a, %a.var
// CHECK:STDOUT:   %p.ref: ref %.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc14_15: %.2 = bind_value %p.ref
// CHECK:STDOUT:   %.loc14_14.1: ref %T = deref %.loc14_15
// CHECK:STDOUT:   %.loc14_14.2: %T = bind_value %.loc14_14.1
// CHECK:STDOUT:   assign %a.var, %.loc14_14.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:

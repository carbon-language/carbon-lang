// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/basic.carbon

class Class(T:! type) {
  fn GetAddr[addr self: Self*]() -> T* {
    return &self->k;
  }

  // TODO: Should this work? T is not necessarily copyable.
  fn GetValue[self: Self]() -> T {
    return self.k;
  }

  var k: T;
}


// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = ptr_type Class [template]
// CHECK:STDOUT:   %.3: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %GetAddr: type = fn_type @GetAddr [template]
// CHECK:STDOUT:   %struct.2: GetAddr = struct_value () [template]
// CHECK:STDOUT:   %GetValue: type = fn_type @GetValue [template]
// CHECK:STDOUT:   %struct.3: GetValue = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.k: T} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type {.k: T} [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %GetAddr.decl: GetAddr = fn_decl @GetAddr [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class.2 [template = constants.%Class.2]
// CHECK:STDOUT:     %.loc12_29: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc12_19.1: Class* = param self
// CHECK:STDOUT:     %self.loc12_19.3: Class* = bind_name self, %self.loc12_19.1
// CHECK:STDOUT:     %.loc12_14: Class* = addr_pattern %self.loc12_19.3
// CHECK:STDOUT:     %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc12_38: type = ptr_type T [symbolic = constants.%.3]
// CHECK:STDOUT:     %return.var.loc12: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GetValue.decl: GetValue = fn_decl @GetValue [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class.2 [template = constants.%Class.2]
// CHECK:STDOUT:     %self.loc17_15.1: Class = param self
// CHECK:STDOUT:     %self.loc17_15.2: Class = bind_name self, %self.loc17_15.1
// CHECK:STDOUT:     %T.ref.loc17: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %return.var.loc17: ref T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref.loc21: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc21: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .GetAddr = %GetAddr.decl
// CHECK:STDOUT:   .GetValue = %GetValue.decl
// CHECK:STDOUT:   .k = %.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetAddr[addr @Class.%self.loc12_19.3: Class*]() -> T* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class* = name_ref self, @Class.%self.loc12_19.3
// CHECK:STDOUT:   %.loc13_17.1: ref Class = deref %self.ref
// CHECK:STDOUT:   %k.ref: <unbound element of class Class> = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:   %.loc13_17.2: ref T = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:   %.loc13_12: T* = addr_of %.loc13_17.2
// CHECK:STDOUT:   return %.loc13_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetValue[@Class.%self.loc17_15.2: Class]() -> T {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, @Class.%self.loc17_15.2
// CHECK:STDOUT:   %k.ref: <unbound element of class Class> = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:   %.loc18_16.1: ref T = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc18_16.2: T = bind_value %.loc18_16.1
// CHECK:STDOUT:   return %.loc18_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

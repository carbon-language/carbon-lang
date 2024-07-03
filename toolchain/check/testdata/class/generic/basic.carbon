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
    // CHECK:STDERR: basic.carbon:[[@LINE+4]]:5: ERROR: Cannot implicitly convert from `T*` to `T*`.
    // CHECK:STDERR:     return &self->k;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return &self->k;
  }

  // TODO: Should this work? T is not necessarily copyable.
  fn GetValue[self: Self]() -> T {
    // CHECK:STDERR: basic.carbon:[[@LINE+7]]:5: ERROR: Cannot implicitly convert from `T` to `T`.
    // CHECK:STDERR:     return self.k;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: basic.carbon:[[@LINE+3]]:12: ERROR: Cannot implicitly convert from `Class` to `Class`.
    // CHECK:STDERR:     return self.k;
    // CHECK:STDERR:            ^~~~~~
    return self.k;
  }

  var k: T;
}


// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, (%T) [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %Class.2 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type file.%T.loc11_13.2 (%T) [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr [template]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [template]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue [template]
// CHECK:STDOUT:   %GetValue: %GetValue.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.2, file.%T.loc11_13.2 (%T) [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.k: file.%T.loc11_13.2 (%T)} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %GetAddr.decl: %GetAddr.type = fn_decl @GetAddr [template = constants.%GetAddr] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class.2 [symbolic = %Self.ref.loc12 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc12_29: type = ptr_type %Class.2 [symbolic = %.loc12_29 (constants.%.2)]
// CHECK:STDOUT:     %self.loc12_19.1: @Class.%.loc12_29 (%.2) = param self
// CHECK:STDOUT:     %self.loc12_19.3: @Class.%.loc12_29 (%.2) = bind_name self, %self.loc12_19.1
// CHECK:STDOUT:     %.loc12_14: @Class.%.loc12_29 (%.2) = addr_pattern %self.loc12_19.3
// CHECK:STDOUT:     %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12 (constants.%T)]
// CHECK:STDOUT:     %.loc12_38: type = ptr_type file.%T.loc11_13.2 (%T) [symbolic = %.loc12_38 (constants.%.3)]
// CHECK:STDOUT:     %return.var.loc12: ref @Class.%.loc12_38 (%.3) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GetValue.decl: %GetValue.type = fn_decl @GetValue [template = constants.%GetValue] {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Class.2 [symbolic = %Self.ref.loc21 (constants.%Class.2)]
// CHECK:STDOUT:     %self.loc21_15.1: @Class.%Self.ref.loc21 (%Class.2) = param self
// CHECK:STDOUT:     %self.loc21_15.2: @Class.%Self.ref.loc21 (%Class.2) = bind_name self, %self.loc21_15.1
// CHECK:STDOUT:     %T.ref.loc21: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc21 (constants.%T)]
// CHECK:STDOUT:     %return.var.loc21: ref @Class.%T.ref.loc21 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref.loc32: type = name_ref T, file.%T.loc11_13.2 [symbolic = file.%T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %.loc32: %.4 = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .GetAddr = %GetAddr.decl
// CHECK:STDOUT:   .GetValue = %GetValue.decl
// CHECK:STDOUT:   .k = %.loc32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetAddr[addr @Class.%self.loc12_19.3: @Class.%.loc12_29 (%.2)]() -> @Class.%.loc12_38 (%.3)
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: @Class.%.loc12_29 (%.2) = name_ref self, @Class.%self.loc12_19.3
// CHECK:STDOUT:   %.loc17_17.1: ref %Class.2 = deref %self.ref
// CHECK:STDOUT:   %k.ref: %.4 = name_ref k, @Class.%.loc32 [template = @Class.%.loc32]
// CHECK:STDOUT:   %.loc17_17.2: ref file.%T.loc11_13.2 (%T) = class_element_access %.loc17_17.1, element0
// CHECK:STDOUT:   %.loc17_12: %.3 = addr_of %.loc17_17.2
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetValue[@Class.%self.loc21_15.2: @Class.%Self.ref.loc21 (%Class.2)]() -> @Class.%T.ref.loc21 (%T)
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: @Class.%Self.ref.loc21 (%Class.2) = name_ref self, @Class.%self.loc21_15.2
// CHECK:STDOUT:   %k.ref: %.4 = name_ref k, @Class.%.loc32 [template = @Class.%.loc32]
// CHECK:STDOUT:   %.loc29: file.%T.loc11_13.2 (%T) = class_element_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

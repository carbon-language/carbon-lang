// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self_type.carbon

class Class {
  fn F[self: Self]() -> i32;
  fn Make() -> Self {
    returned var s: Self;
    s = {.p = &s};
    return var;
  }
  var p: Self*;
}

fn Class.F[self: Self]() -> i32 {
  return (*self.p).F();
}

// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %Class [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, %.2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.p: %.2} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.4 [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc21_12.1: %Class = param self, runtime_param0
// CHECK:STDOUT:     @F.%self: %Class = bind_name self, %self.loc21_12.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_29.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc21_29.2: type = converted %int.make_type_32, %.loc21_29.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc12_8.1: %Class = param self, runtime_param0
// CHECK:STDOUT:     %self.loc12_8.2: %Class = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_25.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc12_25.2: type = converted %int.make_type_32, %.loc12_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.var.loc13: ref %Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref.loc18: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc18_14: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_8: %.3 = field_decl p, element0 [template]
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .p = %.loc18_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, @Class.%.loc18_8 [template = @Class.%.loc18_8]
// CHECK:STDOUT:   %.loc22_16.1: ref %.2 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc22_16.2: %.2 = bind_value %.loc22_16.1
// CHECK:STDOUT:   %.loc22_11.1: ref %Class = deref %.loc22_16.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc22_19: <bound method> = bound_method %.loc22_11.1, %F.ref
// CHECK:STDOUT:   %.loc22_11.2: %Class = bind_value %.loc22_11.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc22_19(%.loc22_11.2)
// CHECK:STDOUT:   %.loc22_23.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_23.2: i32 = converted %F.call, %.loc22_23.1
// CHECK:STDOUT:   return %.loc22_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> @Class.%return.var.loc13: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %s: ref %Class = bind_name s, @Class.%return.var.loc13
// CHECK:STDOUT:   %s.ref.loc15_5: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %s.ref.loc15_16: ref %Class = name_ref s, %s
// CHECK:STDOUT:   %.loc15_15: %.2 = addr_of %s.ref.loc15_16
// CHECK:STDOUT:   %.loc15_17.1: %.4 = struct_literal (%.loc15_15)
// CHECK:STDOUT:   %.loc15_17.2: ref %.2 = class_element_access %s.ref.loc15_5, element0
// CHECK:STDOUT:   %.loc15_17.3: init %.2 = initialize_from %.loc15_15 to %.loc15_17.2
// CHECK:STDOUT:   %.loc15_17.4: init %Class = class_init (%.loc15_17.3), %s.ref.loc15_5
// CHECK:STDOUT:   %.loc15_7: init %Class = converted %.loc15_17.1, %.loc15_17.4
// CHECK:STDOUT:   assign %s.ref.loc15_5, %.loc15_7
// CHECK:STDOUT:   return %s to @Class.%return.var.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:

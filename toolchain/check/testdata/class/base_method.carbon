// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method.carbon

base class Base {
  var a: i32;

  fn F[addr self: Self*]();
}

fn Base.F[addr self: Self*]() {
  (*self).a = 1;
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  (*p).F();
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.8 [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc17_26: type = ptr_type %Base [template = constants.%.3]
// CHECK:STDOUT:     %self.loc17_16.1: %.3 = param self, runtime_param0
// CHECK:STDOUT:     @F.%self: %.3 = bind_name self, %self.loc17_16.1
// CHECK:STDOUT:     @F.%.loc17: %.3 = addr_pattern @F.%self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc25: type = ptr_type %Derived [template = constants.%.9]
// CHECK:STDOUT:     %p.loc25_9.1: %.9 = param p, runtime_param0
// CHECK:STDOUT:     @Call.%p: %.9 = bind_name p, %p.loc25_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc14_23: type = ptr_type %Base [template = constants.%.3]
// CHECK:STDOUT:     %self.loc14_13.1: %.3 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc14_13.3: %.3 = bind_name self, %self.loc14_13.1
// CHECK:STDOUT:     %.loc14_8: %.3 = addr_pattern %self.loc14_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc22: %.7 = base_decl %Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self: %.3]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.3 = name_ref self, %self
// CHECK:STDOUT:   %.loc18_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc18_10: ref i32 = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %.loc18_15: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc18_10, %.loc18_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: %.9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_4.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Base.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc26_7: <bound method> = bound_method %.loc26_4.1, %F.ref
// CHECK:STDOUT:   %.loc26_4.2: %.9 = addr_of %.loc26_4.1
// CHECK:STDOUT:   %.loc26_9.1: ref %Derived = deref %.loc26_4.2
// CHECK:STDOUT:   %.loc26_9.2: ref %Base = class_element_access %.loc26_9.1, element0
// CHECK:STDOUT:   %.loc26_9.3: %.3 = addr_of %.loc26_9.2
// CHECK:STDOUT:   %.loc26_9.4: %.3 = converted %.loc26_4.2, %.loc26_9.3
// CHECK:STDOUT:   %F.call: init %.1 = call %.loc26_7(%.loc26_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

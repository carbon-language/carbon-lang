// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self_conversion.carbon

base class Base {
  var a: i32;
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> i32;
  fn AddrSelfBase[addr self: Base*]();
}

fn Derived.SelfBase[self: Base]() -> i32 {
  return self.a;
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  (*self).a = 1;
}

fn Call(p: Derived*) -> i32 {
  (*p).AddrSelfBase();
  return (*p).SelfBase();
}

// CHECK:STDOUT: --- self_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %SelfBase.type: type = fn_type @SelfBase [template]
// CHECK:STDOUT:   %SelfBase: %SelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %Base [template]
// CHECK:STDOUT:   %AddrSelfBase.type: type = fn_type @AddrSelfBase [template]
// CHECK:STDOUT:   %AddrSelfBase: %AddrSelfBase.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.4} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.7 [template]
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
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [template = constants.%SelfBase] {
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc22_21.1: %Base = param self, runtime_param0
// CHECK:STDOUT:     @SelfBase.%self: %Base = bind_name self, %self.loc22_21.1
// CHECK:STDOUT:     %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_38.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:     %.loc22_38.2: type = converted %int.make_type_32.loc22, %.loc22_38.1 [template = i32]
// CHECK:STDOUT:     @SelfBase.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %Base.ref.loc26: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc26_40: type = ptr_type %Base [template = constants.%.6]
// CHECK:STDOUT:     %self.loc26_30.1: %.6 = param self, runtime_param0
// CHECK:STDOUT:     @AddrSelfBase.%self: %.6 = bind_name self, %self.loc26_30.1
// CHECK:STDOUT:     @AddrSelfBase.%.loc26: %.6 = addr_pattern @AddrSelfBase.%self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc30_19: type = ptr_type %Derived [template = constants.%.9]
// CHECK:STDOUT:     %p.loc30_9.1: %.9 = param p, runtime_param0
// CHECK:STDOUT:     @Call.%p: %.9 = bind_name p, %p.loc30_9.1
// CHECK:STDOUT:     %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc30_25.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:     %.loc30_25.2: type = converted %int.make_type_32.loc30, %.loc30_25.1 [template = i32]
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref.loc16: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc16: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [template = constants.%SelfBase] {
// CHECK:STDOUT:     %Base.ref.loc18: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc18_15.1: %Base = param self, runtime_param0
// CHECK:STDOUT:     %self.loc18_15.2: %Base = bind_name self, %self.loc18_15.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_32.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc18_32.2: type = converted %int.make_type_32, %.loc18_32.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [template = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %Base.ref.loc19: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc19_34: type = ptr_type %Base [template = constants.%.6]
// CHECK:STDOUT:     %self.loc19_24.1: %.6 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc19_24.3: %.6 = bind_name self, %self.loc19_24.1
// CHECK:STDOUT:     %.loc19_19: %.6 = addr_pattern %self.loc19_24.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase[%self: %Base]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc23_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc23_14.2: i32 = bind_value %.loc23_14.1
// CHECK:STDOUT:   return %.loc23_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase[addr %self: %.6]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.6 = name_ref self, %self
// CHECK:STDOUT:   %.loc27_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, @Base.%.loc12_8 [template = @Base.%.loc12_8]
// CHECK:STDOUT:   %.loc27_10: ref i32 = class_element_access %.loc27_4, element0
// CHECK:STDOUT:   %.loc27_15: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   assign %.loc27_10, %.loc27_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: %.9) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc31: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_4.1: ref %Derived = deref %p.ref.loc31
// CHECK:STDOUT:   %AddrSelfBase.ref: %AddrSelfBase.type = name_ref AddrSelfBase, @Derived.%AddrSelfBase.decl [template = constants.%AddrSelfBase]
// CHECK:STDOUT:   %.loc31_7: <bound method> = bound_method %.loc31_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %.loc31_4.2: %.9 = addr_of %.loc31_4.1
// CHECK:STDOUT:   %.loc31_20.1: ref %Derived = deref %.loc31_4.2
// CHECK:STDOUT:   %.loc31_20.2: ref %Base = class_element_access %.loc31_20.1, element0
// CHECK:STDOUT:   %.loc31_20.3: %.6 = addr_of %.loc31_20.2
// CHECK:STDOUT:   %.loc31_20.4: %.6 = converted %.loc31_4.2, %.loc31_20.3
// CHECK:STDOUT:   %AddrSelfBase.call: init %.1 = call %.loc31_7(%.loc31_20.4)
// CHECK:STDOUT:   %p.ref.loc32: %.9 = name_ref p, %p
// CHECK:STDOUT:   %.loc32_11: ref %Derived = deref %p.ref.loc32
// CHECK:STDOUT:   %SelfBase.ref: %SelfBase.type = name_ref SelfBase, @Derived.%SelfBase.decl [template = constants.%SelfBase]
// CHECK:STDOUT:   %.loc32_14: <bound method> = bound_method %.loc32_11, %SelfBase.ref
// CHECK:STDOUT:   %.loc32_23.1: ref %Base = class_element_access %.loc32_11, element0
// CHECK:STDOUT:   %.loc32_23.2: ref %Base = converted %.loc32_11, %.loc32_23.1
// CHECK:STDOUT:   %.loc32_23.3: %Base = bind_value %.loc32_23.2
// CHECK:STDOUT:   %SelfBase.call: init i32 = call %.loc32_14(%.loc32_23.3)
// CHECK:STDOUT:   %.loc32_25.1: i32 = value_of_initializer %SelfBase.call
// CHECK:STDOUT:   %.loc32_25.2: i32 = converted %SelfBase.call, %.loc32_25.1
// CHECK:STDOUT:   return %.loc32_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

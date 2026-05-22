// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial/init.carbon

// --- base.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn Make() -> partial Self {
    return {};
  }
}

class Derived {
  extend base: Base;
  fn Make() -> Self {
    //@dump-sem-ir-begin
    return {.base = Base.Make()};
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %.f66: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %Base.Make.type: type = fn_type @Base.Make [concrete]
// CHECK:STDOUT:   %Base.Make: %Base.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %struct_type.base.b7a: type = struct_type {.base: %.f66} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc11
// CHECK:STDOUT:   .Make = %Derived.Make.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.Make() -> out %return.param: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %Make.ref: %Base.Make.type = name_ref Make, @Base.%Base.Make.decl [concrete = constants.%Base.Make]
// CHECK:STDOUT:   %.loc14_32.1: ref %.f66 = class_element_access %return.param, element0
// CHECK:STDOUT:   %Base.Make.call: init %.f66 to %.loc14_32.1 = call %Make.ref()
// CHECK:STDOUT:   %.loc14_32.2: %struct_type.base.b7a = struct_literal (%Base.Make.call)
// CHECK:STDOUT:   %.loc14_32.3: init %Base = as_compatible %Base.Make.call
// CHECK:STDOUT:   %.loc14_32.4: init %Derived to %return.param = class_init (%.loc14_32.3)
// CHECK:STDOUT:   %.loc14_33: init %Derived = converted %.loc14_32.2, %.loc14_32.4
// CHECK:STDOUT:   return %.loc14_33 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:

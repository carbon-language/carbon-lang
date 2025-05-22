// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  var a: i32;
  var b: i32;
}

//@dump-sem-ir-begin
class SomeClassAdapter {
  adapt SomeClass;
}

class StructAdapter {
  adapt {.a: i32, .b: i32};
}
//@dump-sem-ir-end

// --- fail_not_extend.carbon

library "[[@TEST_NAME]]";

class Adapted {
  fn F();
}

class AdaptNotExtend {
  adapt Adapted;
}

fn F(a: AdaptNotExtend) {
  // `Adapted` is not extended, so lookup for `F` finds nothing.
  // CHECK:STDERR: fail_not_extend.carbon:[[@LINE+4]]:3: error: member name `F` not found in `AdaptNotExtend` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  a.F();
}

// --- fail_misplaced.carbon

fn F() {
  // CHECK:STDERR: fail_misplaced.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt i32;
}

interface I {
  // CHECK:STDERR: fail_misplaced.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt i32;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt i32;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [concrete = constants.%StructAdapter] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %int_32.loc15_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc15_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   adapt_decl %struct_type.a.b [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT: }
// CHECK:STDOUT:

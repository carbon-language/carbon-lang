// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl.carbon

// --- extend_impl.carbon

library "[[@TEST_NAME]]";

interface HasF {
  fn F();
}

//@dump-sem-ir-begin
class C {
  extend impl as HasF {
    fn F() {}
  }
}
//@dump-sem-ir-end

fn G(c: C) {
  C.F();
  c.F();
}

// --- fail_extend_impl_nonexistent.carbon

library "[[@TEST_NAME]]";

interface I {}

class C {
  // CHECK:STDERR: fail_extend_impl_nonexistent.carbon:[[@LINE+4]]:15: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   extend impl nonexistent as I {}
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  extend impl nonexistent as I {}

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the class scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- fail_impl_nonexistent.carbon

library "[[@TEST_NAME]]";

interface I {}

class C {
  // CHECK:STDERR: fail_impl_nonexistent.carbon:[[@LINE+4]]:8: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   impl nonexistent as I {}
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  impl nonexistent as I {}

  fn F() {
    // The name lookup error still happens, since the `require` is not `extend`.
    // CHECK:STDERR: fail_impl_nonexistent.carbon:[[@LINE+4]]:5: error: member name `A` not found in `C` [MemberNameNotFoundInInstScope]
    // CHECK:STDERR:     Self.A;
    // CHECK:STDERR:     ^~~~~~
    // CHECK:STDERR:
    Self.A;
  }
}

// --- fail_extend_impl_nonexistent_pointer.carbon

library "[[@TEST_NAME]]";

interface I {}

class C {
  // CHECK:STDERR: fail_extend_impl_nonexistent_pointer.carbon:[[@LINE+4]]:15: error: name `nonexistent` not found [NameNotFound]
  // CHECK:STDERR:   extend impl nonexistent* as I {}
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  extend impl nonexistent* as I {}

  fn F() {
    // The erroneous self-type for an `extend` causes the error to be propagated
    // into the class scope, which prevents errors if we fail to find a name
    // in that scope.
    Self.A;
  }
}

// --- fail_extend_impl_nonexistent_outside_class.carbon

library "[[@TEST_NAME]]";

interface I {}
// CHECK:STDERR: fail_extend_impl_nonexistent_outside_class.carbon:[[@LINE+4]]:13: error: name `nonexistent` not found [NameNotFound]
// CHECK:STDERR: extend impl nonexistent* as I {}
// CHECK:STDERR:             ^~~~~~~~~~~
// CHECK:STDERR:
extend impl nonexistent* as I {}

// CHECK:STDOUT: --- extend_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.ref {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {} {}
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   extend %HasF.ref
// CHECK:STDOUT:   witness = %HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: type = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @C.as.HasF.impl.%HasF.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

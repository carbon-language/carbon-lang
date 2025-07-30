// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/basic.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

interface Simple {
  fn F();
}

class C {}

//@dump-sem-ir-begin
impl C as Simple {
  fn F() {}
}
//@dump-sem-ir-end

// --- fail_invalid_impl.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Op[self: Self]();
}

class C {}

// This produces an invalid impl.
// CHECK:STDERR: fail_invalid_impl.carbon:[[@LINE+4]]:6: error: name `Unknown` not found [NameNotFound]
// CHECK:STDERR: impl Unknown as I {
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
impl Unknown as I {
  fn Op[self: Self]() {}
}

// --- fail_import_invalid_impl.carbon

library "[[@TEST_NAME]]";
import library "invalid_impl";

fn F() {
  // This impl doesn't exist, but it still tests that we can ignore the
  // `impl Unknown as I` on import.
  // CHECK:STDERR: fail_import_invalid_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `I` in type `type` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   C.(I.Op)();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  C.(I.Op)();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness file.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.Simple.impl.F.type: type = fn_type @C.as.Simple.impl.F [concrete]
// CHECK:STDOUT:   %C.as.Simple.impl.F: %C.as.Simple.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.Simple.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (@C.as.Simple.impl.%C.as.Simple.impl.F.decl), @C.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Simple.impl: %C.ref as %Simple.ref {
// CHECK:STDOUT:   %C.as.Simple.impl.F.decl: %C.as.Simple.impl.F.type = fn_decl @C.as.Simple.impl.F [concrete = constants.%C.as.Simple.impl.F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.Simple.impl.F.decl
// CHECK:STDOUT:   witness = file.%Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Simple.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

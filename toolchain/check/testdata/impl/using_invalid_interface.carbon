// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/using_invalid_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/using_invalid_interface.carbon

// --- fail_invalid_interface.carbon
library "[[@TEST_NAME]]";

interface I {}

class ArgToI(T: I) {}

interface IMisuse {
  let Arg: type;
  // CHECK:STDERR:        ╭─┤ fail_invalid_interface.carbon:[[@LINE+13]]:18
  // CHECK:STDERR:        │ class ArgToI(T: I) {}
  // CHECK:STDERR:        ·              ──┬─
  // CHECK:STDERR:        ·                ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                   │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   fn Op(self) -> ArgToI(Arg);
  // CHECK:STDERR:        ·                  ┳━━━━━━━━━━
  // CHECK:STDERR:        ·                  ┗━━┫ error: cannot convert type `Arg` into type implementing
  // CHECK:STDERR:        ·                     ┃        `I`
  // CHECK:STDERR:        ·                     ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  fn Op(self) -> ArgToI(Arg);
}

// --- fail_misusing_invalid_interface.carbon
library "[[@TEST_NAME]]";
// This import boundary is significant.
import library "invalid_interface";

class C {
  // CHECK:STDERR:        ╭─┤ fail_misusing_invalid_interface.carbon:[[@LINE+6]]:32
  // CHECK:STDERR: ->     │   impl as IMisuse where .Arg = T {
  // CHECK:STDERR:        ·                                ┳
  // CHECK:STDERR:        ·                                ┗━━┫ error: name `T` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl as IMisuse where .Arg = T {
    // CHECK:STDERR:        ╭─┤ fail_misusing_invalid_interface.carbon:[[@LINE+6]]:34
    // CHECK:STDERR: ->     │     fn Op(unused self) -> ArgToI(T) {
    // CHECK:STDERR:        ·                                  ┳
    // CHECK:STDERR:        ·                                  ┗━━┫ error: name `T` not found [NameNotFound]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    fn Op(unused self) -> ArgToI(T) {
      return {};
    }
  }
}

// CHECK:STDERR:        ╭─┤ fail_misusing_invalid_interface.carbon:[[@LINE+13]]:28
// CHECK:STDERR: ->     │ fn F(generic T: IMisuse) { T.Op(); }
// CHECK:STDERR:        ·                            ┳━━━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: 0 arguments passed to function expecting
// CHECK:STDERR:        ·                               ┃        1 argument
// CHECK:STDERR:        ·                               ┃        [CallArgCountMismatch]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_invalid_interface.carbon
// CHECK:STDERR:        │   fn Op(self) -> ArgToI(Arg);
// CHECK:STDERR:        ·   ─────────────┬─────────────
// CHECK:STDERR:        ·                ╰──┤ calling function declared here [InCallToEntity]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(generic T: IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR:        ╭─┤ fail_misusing_invalid_interface.carbon:[[@LINE+15]]:3
  // CHECK:STDERR:        │ fn F(generic T: IMisuse) { T.Op(); }
  // CHECK:STDERR:        ·              ─────┬────
  // CHECK:STDERR:        ·                   ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                      │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(c);
  // CHECK:STDERR:        ·   ┳━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `C` into type
  // CHECK:STDERR:        ·   │  ┃        implementing `IMisuse`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `C` does not implement interface `Core.ImplicitAs(IMisuse)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(c);
}

// --- fail_using_invalid_interface.carbon
library "[[@TEST_NAME]]";
// This import boundary is significant.
import library "invalid_interface";

class ArgC {}

class C {
  impl as IMisuse where .Arg = ArgC {
    // CHECK:STDERR:        ╭─┤ fail_using_invalid_interface.carbon:[[@LINE+14]]:27
    // CHECK:STDERR: ->     │     fn Op(unused self) -> ArgToI(ArgC) {
    // CHECK:STDERR:        ·                           ┳━━━━━━━━━━━
    // CHECK:STDERR:        ·                           ┗━━┫ error: cannot convert type `ArgC` into type
    // CHECK:STDERR:        ·                              ┃        implementing `I`
    // CHECK:STDERR:        ·                              ┃        [ConversionFailureTypeToFacet]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR:        ├─┤ fail_invalid_interface.carbon
    // CHECK:STDERR:        │ class ArgToI(T: I) {}
    // CHECK:STDERR:        ·              ──┬─
    // CHECK:STDERR:        ·                ╰──┤ initializing generic parameter `T` declared here
    // CHECK:STDERR:        ·                   │ [InitializingGenericParam]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    fn Op(unused self) -> ArgToI(ArgC) {
      return {};
    }
  }
}

// CHECK:STDERR:        ╭─┤ fail_using_invalid_interface.carbon:[[@LINE+13]]:28
// CHECK:STDERR: ->     │ fn F(generic T: IMisuse) { T.Op(); }
// CHECK:STDERR:        ·                            ┳━━━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: 0 arguments passed to function expecting
// CHECK:STDERR:        ·                               ┃        1 argument
// CHECK:STDERR:        ·                               ┃        [CallArgCountMismatch]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ fail_invalid_interface.carbon
// CHECK:STDERR:        │   fn Op(self) -> ArgToI(Arg);
// CHECK:STDERR:        ·   ─────────────┬─────────────
// CHECK:STDERR:        ·                ╰──┤ calling function declared here [InCallToEntity]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(generic T: IMisuse) { T.Op(); }

fn G() {
  var c: C;
  // CHECK:STDERR:        ╭─┤ fail_using_invalid_interface.carbon:[[@LINE+15]]:3
  // CHECK:STDERR:        │ fn F(generic T: IMisuse) { T.Op(); }
  // CHECK:STDERR:        ·              ─────┬────
  // CHECK:STDERR:        ·                   ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                      │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(c);
  // CHECK:STDERR:        ·   ┳━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `C` into type
  // CHECK:STDERR:        ·   │  ┃        implementing `IMisuse`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `C` does not implement interface `Core.ImplicitAs(IMisuse)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(c);
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

interface I;

// `N.F` uses `N.I` and not `package.I`.
namespace N;
interface N.I {}
fn N.F(x: I*);

fn TestCall(x: N.I*) {
  // `N.F` accepts an `N.I*` not a `package.I*`.
  N.F(x);
}

// --- poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F(x: I*);

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:11: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: I*);
// CHECK:STDERR:           ^
fn N.F(x: I*);

// Failure: N.I declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I {}

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x: I*);

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:12: error: member name `I` not found in `N` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn N.F2(x: N.I*);
// CHECK:STDERR:            ^~~
// CHECK:STDERR:
fn N.F2(x: N.I*);

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

interface I;

namespace N;
// Use `package.I` and poison `N.I`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:12: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: I*);
// CHECK:STDERR:            ^
fn N.F1(x: I*);

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I;
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I;

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:12: error: member name `I` not found in `N` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn N.F2(x: N.I*);
// CHECK:STDERR:            ^~~
// CHECK:STDERR:
fn N.F2(x: N.I*);

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

interface I;
namespace N;

// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.I = I;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.I = I;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias N.I = I;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

interface I1;

interface I2 {
  interface I3 {
    interface I4 {
      // Use `package.I1` and poison:
      // * `I2.I1`
      // * `I2.I3.I1`
      // * `I2.I3.I4.I1`
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:15: error: name `I1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: I1*);
      // CHECK:STDERR:               ^~
      fn F(x: I1*);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:17: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       interface I1;
      // CHECK:STDERR:                 ^~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:15: error: name `I1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: I1*);
      // CHECK:STDERR:               ^~
      interface I1;
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:15: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     interface I1;
    // CHECK:STDERR:               ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-15]]:15: error: name `I1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:       fn F(x: I1*);
    // CHECK:STDERR:               ^~
    interface I1;
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   interface I1;
  // CHECK:STDERR:             ^~
  // CHECK:STDERR:
  interface I1;
}

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
interface N.I;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N.I` was poisoned in the api.
interface N.I;

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// `package.I` and `N.I` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: error: name `I` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x: I*);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:11: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: I*);
// CHECK:STDERR:           ^
fn N.F(x: I*);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface I;
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:11: error: name `I` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: I*);
// CHECK:STDERR:           ^
interface I;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:13: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: interface N.I;
// CHECK:STDERR:             ^
// CHECK:STDERR:
interface N.I;

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F() {
  interface I1 {}

  class C {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:12: error: name `I1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     var v: I1;
    // CHECK:STDERR:            ^~
    var v: I1;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:15: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     interface I1;
    // CHECK:STDERR:               ^~
    // CHECK:STDERR:
    interface I1;
  }
}

// --- using_poisoned_name_in_impl.carbon

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x: I*);

class N.C {
  extend impl as I {
  }
}

// CHECK:STDOUT: --- no_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [concrete]
// CHECK:STDOUT:   %I.type.4da: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT:   %Self: %I.type.4da = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type.4da [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestCall.type: type = fn_type @TestCall [concrete]
// CHECK:STDOUT:   %TestCall: %TestCall.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .TestCall = %TestCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc4: type = interface_decl @I.1 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl.loc8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc8: type = interface_decl @I.2 [concrete = constants.%I.type.4da] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc9: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl.loc8 [concrete = constants.%I.type.4da]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type.4da [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestCall.decl: %TestCall.type = fn_decl @TestCall [concrete = constants.%TestCall] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %N.ref.loc11: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl.loc8 [concrete = constants.%I.type.4da]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type.4da [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: %I.type.4da = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCall(%x.param_patt: %ptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref.loc13: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %x.ref: %ptr = name_ref x, %x
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_declare_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type.733 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.4da: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT:   %Self: %I.type.4da = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc4: type = interface_decl @I.1 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type.733 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc18: type = interface_decl @I.2 [concrete = constants.%I.type.4da] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: %I.type.4da = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %N.ref: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:       %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_declaration_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type.733 [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.4da: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc4: type = interface_decl @I.1 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl.loc4 [concrete = constants.%I.type.733]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type.733 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc17: type = interface_decl @I.2 [concrete = constants.%I.type.4da] {} {}
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %N.ref: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:       %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %I: type = bind_alias I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_multiple_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I1.type.80c: type = facet_type <@I1.1> [concrete]
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %Self.c7b: %I2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I3.type.07a: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %I3.type.b2f: type = facet_type <@I3, @I3(%Self.c7b)> [symbolic]
// CHECK:STDOUT:   %Self.60c: %I3.type.b2f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I4.type.451: type = facet_type <@I4> [concrete]
// CHECK:STDOUT:   %I4.type.78e: type = facet_type <@I4, @I4(%Self.c7b, %Self.60c)> [symbolic]
// CHECK:STDOUT:   %Self.a4d: %I4.type.78e = bind_symbolic_name Self, 2 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %I1.type.80c [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I4(%Self.c7b, %Self.60c) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I4.assoc_type: type = assoc_entity_type %I4.type.78e [symbolic]
// CHECK:STDOUT:   %assoc0: %I4.assoc_type = assoc_entity element0, @I4.%F.decl [symbolic]
// CHECK:STDOUT:   %I1.type.f4e: type = facet_type <@I1.2> [concrete]
// CHECK:STDOUT:   %I1.type.575: type = facet_type <@I1.3> [concrete]
// CHECK:STDOUT:   %I1.type.e5b: type = facet_type <@I1.4> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I1 = %I1.decl
// CHECK:STDOUT:     .I2 = %I2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I1.decl: type = interface_decl @I1.1 [concrete = constants.%I1.type.80c] {} {}
// CHECK:STDOUT:   %I2.decl: type = interface_decl @I2 [concrete = constants.%I2.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I1.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.c7b]
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [concrete = constants.%I3.type.07a] {} {}
// CHECK:STDOUT:   %I1.decl: type = interface_decl @I1.4 [concrete = constants.%I1.type.e5b] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I3 = %I3.decl
// CHECK:STDOUT:   .I1 = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I3(@I2.%Self: %I2.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %I2.type = bind_symbolic_name Self, 0 [symbolic = %Self.2 (constants.%Self.c7b)]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3, @I3(%Self.2)> [symbolic = %I3.type (constants.%I3.type.b2f)]
// CHECK:STDOUT:   %Self.3: %I3.type.b2f = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.60c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I3.%I3.type (%I3.type.b2f) = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.60c)]
// CHECK:STDOUT:     %I4.decl: type = interface_decl @I4 [concrete = constants.%I4.type.451] {} {}
// CHECK:STDOUT:     %I1.decl: type = interface_decl @I1.3 [concrete = constants.%I1.type.575] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .I4 = %I4.decl
// CHECK:STDOUT:     .I1 = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I4(@I2.%Self: %I2.type, @I3.%Self.1: @I3.%I3.type (%I3.type.b2f)) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %I2.type = bind_symbolic_name Self, 0 [symbolic = %Self.2 (constants.%Self.c7b)]
// CHECK:STDOUT:   %Self.3: %I3.type.b2f = bind_symbolic_name Self, 1 [symbolic = %Self.3 (constants.%Self.60c)]
// CHECK:STDOUT:   %I4.type: type = facet_type <@I4, @I4(%Self.2, %Self.3)> [symbolic = %I4.type (constants.%I4.type.78e)]
// CHECK:STDOUT:   %Self.4: %I4.type.78e = bind_symbolic_name Self, 2 [symbolic = %Self.4 (constants.%Self.a4d)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I4(%Self.2, %Self.3) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I4.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %I4.assoc_type: type = assoc_entity_type @I4.%I4.type (%I4.type.78e) [symbolic = %I4.assoc_type (constants.%I4.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc16_19.2: @I4.%I4.assoc_type (%I4.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc16_19.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I4.%I4.type (%I4.type.78e) = bind_symbolic_name Self, 2 [symbolic = %Self.4 (constants.%Self.a4d)]
// CHECK:STDOUT:     %F.decl: @I4.%F.type (%F.type) = fn_decl @F [symbolic = @I4.%F (constants.%F)] {
// CHECK:STDOUT:       %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:       %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:       %.loc16: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:         %I1.ref: type = name_ref I1, file.%I1.decl [concrete = constants.%I1.type.80c]
// CHECK:STDOUT:         %ptr: type = ptr_type %I1.type.80c [concrete = constants.%ptr]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_19.1: @I4.%I4.assoc_type (%I4.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc16_19.2 (constants.%assoc0)]
// CHECK:STDOUT:     %I1.decl: type = interface_decl @I1.2 [concrete = constants.%I1.type.f4e] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .I1 = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc16_19.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I1.2(@I2.%Self: %I2.type, @I3.%Self.1: @I3.%I3.type (%I3.type.b2f), @I4.%Self.1: @I4.%I4.type (%I4.type.78e)) {
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I1.3(@I2.%Self: %I2.type, @I3.%Self.1: @I3.%I3.type (%I3.type.b2f)) {
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I1.4(@I2.%Self: %I2.type) {
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I2.%Self: %I2.type, @I3.%Self.1: @I3.%I3.type (%I3.type.b2f), @I4.%Self.1: @I4.%I4.type (%I4.type.78e)) {
// CHECK:STDOUT:   fn(%x.param_patt: %ptr);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I3(constants.%Self.c7b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I4(constants.%Self.c7b, constants.%Self.60c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.c7b, constants.%Self.60c, constants.%Self.a4d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1.2(constants.%Self.c7b, constants.%Self.60c, constants.%Self.a4d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I4(%Self.2, %Self.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1.3(constants.%Self.c7b, constants.%Self.60c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I3(%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1.4(constants.%Self.c7b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- ignored_poison_in_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//poison, I, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [concrete] {
// CHECK:STDOUT:     .F = %Main.F
// CHECK:STDOUT:     .I = file.%I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//poison, I, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [concrete] {
// CHECK:STDOUT:     .F = %Main.F
// CHECK:STDOUT:     .I = file.%I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_when_lookup_fails.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [concrete]
// CHECK:STDOUT:   %I.type.4da: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %ptr [concrete = <error>] {
// CHECK:STDOUT:       %I.ref: <error> = name_ref I, <error> [concrete = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc23: type = interface_decl @I.1 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %I.decl.loc28: type = interface_decl @I.2 [concrete = constants.%I.type.4da] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_lexical_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I1.type.06a: type = facet_type <@I1.1> [concrete]
// CHECK:STDOUT:   %Self: %I1.type.06a = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %I1.type.06a [concrete]
// CHECK:STDOUT:   %I1.type.8ea: type = facet_type <@I1.2> [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %I1.type.06a} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I1.1 {
// CHECK:STDOUT:   %Self: %I1.type.06a = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I1.2;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11_10: %C.elem = field_decl v, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc11_5: %C.elem = var_pattern %.loc11_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %I1.decl: type = interface_decl @I1.2 [concrete = constants.%I1.type.8ea] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I1 = <poisoned>
// CHECK:STDOUT:   .v = %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %I1.decl: type = interface_decl @I1.1 [concrete = constants.%I1.type.06a] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- using_poisoned_name_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %I.type [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %ptr = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %ptr = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %ptr = value_param runtime_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       %ptr: type = ptr_type %I.type [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %ptr = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   extend @impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %ptr);
// CHECK:STDOUT:

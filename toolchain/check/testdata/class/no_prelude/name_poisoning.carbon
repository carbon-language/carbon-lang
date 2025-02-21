// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

class C;

// `N.F` uses `N.C` and not `package.C`.
namespace N;
class N.C {}
fn N.F(x: C);

fn TestCall(x: N.C) {
  // `N.F` accepts an `N.C` not a `package.C`.
  N.F(x);
}

// --- poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
fn N.F(x: C);

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
fn N.F(x: C);

// Failure: `N.C` declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
fn N.F1() -> C;

// Use `N.C` which was poisoned and not declared.
// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:14: error: member name `C` not found in `N` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn N.F2() -> N.C;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
fn N.F2() -> N.C;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// Use `package.C` and poison `N.C`.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:12: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F1(x: C);
// CHECK:STDERR:            ^
fn N.F1(x: C);

// Failure: N.C declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// Failure: `N.C` used after declaration failed.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:12: error: member name `C` not found in `N` [MemberNameNotFoundInScope]
// CHECK:STDERR: fn N.F2(x: N.C);
// CHECK:STDERR:            ^~~
// CHECK:STDERR:
fn N.F2(x: N.C);

// --- fail_alias.carbon

library "[[@TEST_NAME]]";

class C;

namespace N;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:13: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N.C = C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias N.C = C;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

class C1;

class C2 {
  class C3 {
    class C4 {
      // Use `package.C1` and poison:
      // * `C2.C1`
      // * `C2.C3.C1`
      // * `C2.C3.C4.C1`
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C1);
      // CHECK:STDERR:               ^~
      fn F(x: C1);

      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:13: note: declared here [NameUseBeforeDeclNote]
      // CHECK:STDERR:       class C1;
      // CHECK:STDERR:             ^~
      // CHECK:STDERR:
      // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
      // CHECK:STDERR:       fn F(x: C1);
      // CHECK:STDERR:               ^~
      class C1;
    }
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C1;
    // CHECK:STDERR:           ^~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-15]]:15: error: name `C1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:       fn F(x: C1);
    // CHECK:STDERR:               ^~
    class C1;
  }
  // CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   class C1;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  class C1;
}

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
class N.C;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N.C` was poisoned in the api.
class N.C {}

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N;
// `package.C` and `N.C` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: error: name `C` not found [NameNotFound]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
fn N.F(x: C);

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:7: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:11: error: name `C` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: fn N.F(x: C);
// CHECK:STDERR:           ^
class C;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:9: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: class N.C;
// CHECK:STDERR:         ^
// CHECK:STDERR:
class N.C;

// --- fail_poison_with_lexical_result.carbon

library "[[@TEST_NAME]]";

fn F() {
  class C1 {}

  class C2 {
    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+3]]:12: error: name `C1` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     var v: C1;
    // CHECK:STDERR:            ^~
    var v: C1;

    // CHECK:STDERR: fail_poison_with_lexical_result.carbon:[[@LINE+4]]:11: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     class C1;
    // CHECK:STDERR:           ^~
    // CHECK:STDERR:
    class C1;
  }
}

// --- fail_declare_data_member_after_poison.carbon

library "[[@TEST_NAME]]";

class C1;

class C2 {
  // Use `package.C1` and poison `C2.C1`.
  // CHECK:STDERR: fail_declare_data_member_after_poison.carbon:[[@LINE+3]]:11: error: name `C1` used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   fn F(x: C1);
  // CHECK:STDERR:           ^~
  fn F(x: C1);

  class C2 {}
  // Failure: `C2.C1` declared after it was poisoned.
  // CHECK:STDERR: fail_declare_data_member_after_poison.carbon:[[@LINE+4]]:7: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   var C1: C2;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var C1: C2;
}

// --- fail_extend_poison_class_members.carbon

library "[[@TEST_NAME]]";

base class B {}

class C {
  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+3]]:16: error: name `B` used before it was declared [NameUseBeforeDecl]
  // CHECK:STDERR:   extend base: B;
  // CHECK:STDERR:                ^
  extend base: B;

  // CHECK:STDERR: fail_extend_poison_class_members.carbon:[[@LINE+4]]:6: note: declared here [NameUseBeforeDeclNote]
  // CHECK:STDERR:   fn B();
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  fn B();
}

// CHECK:STDOUT: --- no_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [concrete]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestCall.type: type = fn_type @TestCall [concrete]
// CHECK:STDOUT:   %TestCall: %TestCall.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .TestCall = %TestCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [concrete = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc8: type = class_decl @C.2 [concrete = constants.%C.9f4] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %C.9f4 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.9f4 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.9f4 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc8 [concrete = constants.%C.9f4]
// CHECK:STDOUT:     %x: %C.9f4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestCall.decl: %TestCall.type = fn_decl @TestCall [concrete = constants.%TestCall] {
// CHECK:STDOUT:     %x.patt: %C.9f4 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.9f4 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.9f4 = value_param runtime_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %C.ref [concrete = constants.%C.9f4] {
// CHECK:STDOUT:       %N.ref.loc11: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl.loc8 [concrete = constants.%C.9f4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %C.9f4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %C.9f4);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCall(%x.param_patt: %C.9f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N.ref.loc13: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %x.ref: %C.9f4 = name_ref x, %x
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_declare_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [concrete = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc18: type = class_decl @C.2 [concrete = constants.%C.9f4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %C.f79);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.ref: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:     %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_declaration_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [concrete]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2 [concrete]
// CHECK:STDOUT:   %F2: %F2.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C.1 [concrete = constants.%C.f79] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt: %C.f79 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C.f79 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C.f79 = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl.loc4 [concrete = constants.%C.f79]
// CHECK:STDOUT:     %x: %C.f79 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc18: type = class_decl @C.2 [concrete = constants.%C.9f4] {} {}
// CHECK:STDOUT:   %F2.decl: %F2.type = fn_decl @F2 [concrete = constants.%F2] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %N.ref: <namespace> = name_ref N, file.%N [concrete = file.%N]
// CHECK:STDOUT:       %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F1(%x.param_patt: %C.f79);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F2(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C: type = bind_alias C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_multiple_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1.26b: type = class_type @C1.1 [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %C3: type = class_type @C3 [concrete]
// CHECK:STDOUT:   %C4: type = class_type @C4 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C1.d8b: type = class_type @C1.2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C1.a31: type = class_type @C1.3 [concrete]
// CHECK:STDOUT:   %C1.f31: type = class_type @C1.4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.1 [concrete = constants.%C1.26b] {} {}
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [concrete = constants.%C3] {} {}
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.4 [concrete = constants.%C1.f31] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT:   .C3 = %C3.decl
// CHECK:STDOUT:   .C1 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %C4.decl: type = class_decl @C4 [concrete = constants.%C4] {} {}
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.3 [concrete = constants.%C1.a31] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT:   .C4 = %C4.decl
// CHECK:STDOUT:   .C1 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %C1.26b = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C1.26b = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C1.26b = value_param runtime_param0
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [concrete = constants.%C1.26b]
// CHECK:STDOUT:     %x: %C1.26b = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.2 [concrete = constants.%C1.d8b] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C4
// CHECK:STDOUT:   .C1 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.2;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.3;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.4;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %C1.26b);
// CHECK:STDOUT:
// CHECK:STDOUT: --- ignored_poison_in_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [concrete] {
// CHECK:STDOUT:     .F = %Main.F
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//poison, C, unloaded
// CHECK:STDOUT:   %Main.N: <namespace> = import_ref Main//poison, N, loaded
// CHECK:STDOUT:   %N: <namespace> = namespace %Main.N, [concrete] {
// CHECK:STDOUT:     .F = %Main.F
// CHECK:STDOUT:     .C = file.%C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .N = imports.%N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_when_lookup_fails.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f79: type = class_type @C.1 [concrete]
// CHECK:STDOUT:   %C.9f4: type = class_type @C.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: <error> = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:     %x: <error> = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc23: type = class_decl @C.1 [concrete = constants.%C.f79] {} {}
// CHECK:STDOUT:   %C.decl.loc28: type = class_decl @C.2 [concrete = constants.%C.9f4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: <error>);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_with_lexical_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C1.138: type = class_type @C1.1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %C2.elem: type = unbound_element_type %C2, %C1.138 [concrete]
// CHECK:STDOUT:   %C1.46c: type = class_type @C1.2 [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %C1.138} [concrete]
// CHECK:STDOUT:   %complete_type.fb7: <witness> = complete_type_witness %struct_type.v [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1.138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %.loc11_10: %C2.elem = field_decl v, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc11_5: %C2.elem = var_pattern %.loc11_10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C2.elem = var <none>
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.2 [concrete = constants.%C1.46c] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete = constants.%complete_type.fb7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT:   .C1 = <poisoned>
// CHECK:STDOUT:   .v = %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1.1 [concrete = constants.%C1.138] {} {}
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_declare_data_member_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %C2.311: type = class_type @C2.1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C2.0a0: type = class_type @C2.2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C2.elem: type = unbound_element_type %C2.311, %C2.0a0 [concrete]
// CHECK:STDOUT:   %struct_type.C1: type = struct_type {.C1: %C2.0a0} [concrete]
// CHECK:STDOUT:   %complete_type.979: <witness> = complete_type_witness %struct_type.C1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [concrete = constants.%C1] {} {}
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2.1 [concrete = constants.%C2.311] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2.1 {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %C1 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C1 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %C1 = value_param runtime_param0
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:     %x: %C1 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2.2 [concrete = constants.%C2.0a0] {} {}
// CHECK:STDOUT:   %.loc19_9: %C2.elem = field_decl C1, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc19_3: %C2.elem = var_pattern %.loc19_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C2.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.C1 [concrete = constants.%complete_type.979]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2.311
// CHECK:STDOUT:   .C1 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .C2 = %C2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2.0a0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param_patt: %C1);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_poison_class_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.fa3: type = class_type @B.2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B.fa3 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B.1 [concrete]
// CHECK:STDOUT:   %B.489: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B.fa3} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B.2 [concrete = constants.%B.fa3] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.fa3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.fa3]
// CHECK:STDOUT:   %.loc10: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B.1 [concrete = constants.%B.489] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.1();
// CHECK:STDOUT:

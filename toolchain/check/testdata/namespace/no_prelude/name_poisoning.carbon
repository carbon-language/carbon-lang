// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/no_prelude/name_poisoning.carbon

// --- no_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// `N2.N3` uses `N2.N1` and not `package.N1`.
namespace N2;
namespace N2.N1;
alias N2.N3 = N1;

class N2.N1.C {}
class N1.C {}
fn TestNamespaces() {
  var x: N2.N1.C;
  var y: N2.N3.C* = &x;
}

// --- poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
alias N2.N3 = N1;

// --- fail_declare_after_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+3]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3 = N1;
// CHECK:STDERR:               ^~
alias N2.N3 = N1;

// Failure: N2.N1 declared after it was poisoned.
// CHECK:STDERR: fail_declare_after_poison.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// --- fail_use_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
alias N2.N3 = N1;

// CHECK:STDERR: fail_use_poison.carbon:[[@LINE+4]]:15: error: member name `N1` not found in `N2` [MemberNameNotFoundInScope]
// CHECK:STDERR: alias N2.N4 = N2.N1;
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
alias N2.N4 = N2.N1;

// --- fail_use_declaration_after_poison.carbon

library "[[@TEST_NAME]]";

namespace N1;

// Use `package.N1` and poison `N2.N1`.
namespace N2;
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+3]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3 = N1;
// CHECK:STDERR:               ^~
alias N2.N3 = N1;

// Failure: `N2.N1` declared after it was poisoned.
// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// CHECK:STDERR: fail_use_declaration_after_poison.carbon:[[@LINE+4]]:15: error: member name `N1` not found in `N2` [MemberNameNotFoundInScope]
// CHECK:STDERR: alias N2.N4 = N2.N1;
// CHECK:STDERR:               ^~~~~
// CHECK:STDERR:
alias N2.N4 = N2.N1;

// --- fail_alias.carbon

namespace N1;

namespace N2;
// CHECK:STDERR: fail_alias.carbon:[[@LINE+7]]:15: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N1 = N1;
// CHECK:STDERR:               ^~
// CHECK:STDERR: fail_alias.carbon:[[@LINE+4]]:10: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: alias N2.N1 = N1;
// CHECK:STDERR:          ^~
// CHECK:STDERR:
alias N2.N1 = N1;

// --- fail_poison_multiple_scopes.carbon

library "[[@TEST_NAME]]";

namespace N1;
namespace N2;
namespace N2.N3;
namespace N2.N3.N4;
// Use `package.N1` and poison:
// * `N2.N1`
// * `N2.N3.N1`
// * `N2.N3.N4.N1`
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+3]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
alias N2.N3.N4.N5 = N1;

// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:20: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N3.N4.N1;
// CHECK:STDERR:                    ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-6]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
namespace N2.N3.N4.N1;
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+7]]:17: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N3.N1;
// CHECK:STDERR:                 ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE-14]]:21: error: name `N1` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N2.N3.N4.N5 = N1;
// CHECK:STDERR:                     ^~
namespace N2.N3.N1;
// CHECK:STDERR: fail_poison_multiple_scopes.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N2.N1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N2.N1;

// --- ignored_poison_in_import.carbon

library "[[@TEST_NAME]]";
import library "poison";

// This doesn't fail.
namespace N2.N1;

// --- poison.impl.carbon

impl library "[[@TEST_NAME]]";

// TODO: #4622 This should fail since `N2.N1` was poisoned in the api.
namespace N2.N1;

// --- fail_poison_when_lookup_fails.carbon

library "[[@TEST_NAME]]";

namespace N1;
// `N1.N3` poisoned when not found.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:15: error: name `N3` not found [NameNotFound]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+3]]:15: error: name `N3` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
alias N1.N2 = N3;

// TODO: We should ideally only produce one diagnostic here.
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+7]]:11: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N3;
// CHECK:STDERR:           ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE-7]]:15: error: name `N3` used before it was declared [NameUseBeforeDecl]
// CHECK:STDERR: alias N1.N2 = N3;
// CHECK:STDERR:               ^~
namespace N3;
// CHECK:STDERR: fail_poison_when_lookup_fails.carbon:[[@LINE+4]]:14: note: declared here [NameUseBeforeDeclNote]
// CHECK:STDERR: namespace N1.N3;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
namespace N1.N3;

// CHECK:STDOUT: --- no_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.2a3: type = class_type @C.1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.0b8: type = class_type @C.2 [concrete]
// CHECK:STDOUT:   %TestNamespaces.type: type = fn_type @TestNamespaces [concrete]
// CHECK:STDOUT:   %TestNamespaces: %TestNamespaces.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.33b: type = ptr_type %C.2a3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc4
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:     .TestNamespaces = %TestNamespaces.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc4: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc8
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc8: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, %N1.loc8 [concrete = %N1.loc8]
// CHECK:STDOUT:   %N3: <namespace> = bind_alias N3, %N1.loc8 [concrete = %N1.loc8]
// CHECK:STDOUT:   %C.decl.loc11: type = class_decl @C.1 [concrete = constants.%C.2a3] {} {}
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C.2 [concrete = constants.%C.0b8] {} {}
// CHECK:STDOUT:   %TestNamespaces.decl: %TestNamespaces.type = fn_decl @TestNamespaces [concrete = constants.%TestNamespaces] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.2a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C.2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C.0b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNamespaces() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %C.2a3 = binding_pattern x
// CHECK:STDOUT:     %.loc14_3: %C.2a3 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %C.2a3 = var x
// CHECK:STDOUT:   %.loc14_15: type = splice_block %C.ref.loc14 [concrete = constants.%C.2a3] {
// CHECK:STDOUT:     %N2.ref.loc14: <namespace> = name_ref N2, file.%N2 [concrete = file.%N2]
// CHECK:STDOUT:     %N1.ref: <namespace> = name_ref N1, file.%N1.loc8 [concrete = file.%N1.loc8]
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl.loc11 [concrete = constants.%C.2a3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %C.2a3 = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %ptr.33b = binding_pattern y
// CHECK:STDOUT:     %.loc15_3: %ptr.33b = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %ptr.33b = var y
// CHECK:STDOUT:   %x.ref: ref %C.2a3 = name_ref x, %x
// CHECK:STDOUT:   %addr: %ptr.33b = addr_of %x.ref
// CHECK:STDOUT:   assign %y.var, %addr
// CHECK:STDOUT:   %.loc15_17: type = splice_block %ptr [concrete = constants.%ptr.33b] {
// CHECK:STDOUT:     %N2.ref.loc15: <namespace> = name_ref N2, file.%N2 [concrete = file.%N2]
// CHECK:STDOUT:     %N3.ref: <namespace> = name_ref N3, file.%N3 [concrete = file.%N1.loc8]
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, file.%C.decl.loc11 [concrete = constants.%C.2a3]
// CHECK:STDOUT:     %ptr: type = ptr_type %C.2a3 [concrete = constants.%ptr.33b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %ptr.33b = bind_name y, %y.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, %N1 [concrete = %N1]
// CHECK:STDOUT:   %N3: <namespace> = bind_alias N3, %N1 [concrete = %N1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_declare_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc4
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc4: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N3: <namespace> = bind_alias N3, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N1.loc18: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:     .N2 = <poisoned>
// CHECK:STDOUT:     .N4 = %N4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref.loc8: <namespace> = name_ref N1, %N1 [concrete = %N1]
// CHECK:STDOUT:   %N3: <namespace> = bind_alias N3, %N1 [concrete = %N1]
// CHECK:STDOUT:   %N2.ref: <namespace> = name_ref N2, %N2 [concrete = %N2]
// CHECK:STDOUT:   %N1.ref.loc14: <error> = name_ref N1, <error> [concrete = <error>]
// CHECK:STDOUT:   %N4: <error> = bind_alias N4, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_declaration_after_poison.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc4
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc4: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:     .N2 = <poisoned>
// CHECK:STDOUT:     .N4 = %N4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref.loc11: <namespace> = name_ref N1, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N3: <namespace> = bind_alias N3, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N1.loc18: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2.ref: <namespace> = name_ref N2, %N2 [concrete = %N2]
// CHECK:STDOUT:   %N1.ref.loc24: <error> = name_ref N1, <error> [concrete = <error>]
// CHECK:STDOUT:   %N4: <error> = bind_alias N4, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc2
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc2: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, %N1.loc2 [concrete = %N1.loc2]
// CHECK:STDOUT:   %N1.loc12: <namespace> = bind_alias N1, %N1.loc2 [concrete = %N1.loc2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_multiple_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1.loc4
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.loc4: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N2: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N3 = %N3
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N3: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N4 = %N4
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N4: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = <poisoned>
// CHECK:STDOUT:     .N5 = %N5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1.ref: <namespace> = name_ref N1, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N5: <namespace> = bind_alias N5, %N1.loc4 [concrete = %N1.loc4]
// CHECK:STDOUT:   %N1.loc24: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N1.loc32: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N1.loc37: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ignored_poison_in_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.N1: <namespace> = import_ref Main//poison, N1, loaded
// CHECK:STDOUT:   %N1: <namespace> = namespace %Main.N1, [concrete] {}
// CHECK:STDOUT:   %Main.N2: <namespace> = import_ref Main//poison, N2, loaded
// CHECK:STDOUT:   %N2: <namespace> = namespace %Main.N2, [concrete] {
// CHECK:STDOUT:     .N3 = %Main.N3
// CHECK:STDOUT:     .N1 = file.%N1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = imports.%N1
// CHECK:STDOUT:     .N2 = imports.%N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- poison.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.N2: <namespace> = import_ref Main//poison, N2, loaded
// CHECK:STDOUT:   %N2: <namespace> = namespace %Main.N2, [concrete] {
// CHECK:STDOUT:     .N3 = %Main.N3
// CHECK:STDOUT:     .N1 = file.%N1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.N1: <namespace> = import_ref Main//poison, N1, loaded
// CHECK:STDOUT:   %N1: <namespace> = namespace %Main.N1, [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N2 = imports.%N2
// CHECK:STDOUT:     .N1 = imports.%N1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <none>
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_poison_when_lookup_fails.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N1 = %N1
// CHECK:STDOUT:     .N3 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N3 = <poisoned>
// CHECK:STDOUT:     .N2 = %N2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N3.ref: <error> = name_ref N3, <error> [concrete = <error>]
// CHECK:STDOUT:   %N2: <error> = bind_alias N2, <error> [concrete = <error>]
// CHECK:STDOUT:   %N3.loc23: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %N3.loc28: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:

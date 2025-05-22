// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/extern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/extern.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- decl.carbon

library "[[@TEST_NAME]]";

class C;

// --- extern_decl.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
extern class C;
//@dump-sem-ir-end

// --- extern_decl_copy.carbon

library "[[@TEST_NAME]]";

extern class C;

// --- def.carbon

library "[[@TEST_NAME]]";

class C {}

// ============================================================================
// Test files
// ============================================================================

// --- fail_decl_fn_in_extern.carbon

library "[[@TEST_NAME]]";

extern class C;
// CHECK:STDERR: fail_decl_fn_in_extern.carbon:[[@LINE+7]]:4: error: cannot declare a member of incomplete class `C` [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR: fn C.F();
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_decl_fn_in_extern.carbon:[[@LINE-4]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.F();

// --- extern_def.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
extern class C {}
//@dump-sem-ir-end

// --- fail_extern_decl_after_extern_decl.carbon

library "[[@TEST_NAME]]";

extern class C;
// CHECK:STDERR: fail_extern_decl_after_extern_decl.carbon:[[@LINE+7]]:1: error: redeclaration of `class C` is redundant [RedeclRedundant]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_extern_decl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_decl_after_extern_decl.carbon

library "[[@TEST_NAME]]";

extern class C;
// CHECK:STDERR: fail_decl_after_extern_decl.carbon:[[@LINE+7]]:1: error: redeclaration of `class C` is redundant [RedeclRedundant]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_after_extern_decl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
class C;

// --- fail_extern_member_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_extern_member_class.carbon:[[@LINE+4]]:3: error: `extern` not allowed; requires file or namespace scope [ModifierExternNotAllowed]
  // CHECK:STDERR:   extern class D;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern class D;
}

// --- fail_def_after_extern_decl.carbon

library "[[@TEST_NAME]]";

extern class C;
// CHECK:STDERR: fail_def_after_extern_decl.carbon:[[@LINE+7]]:1: error: redeclarations of `class C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_def_after_extern_decl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
class C {}

// --- fail_extern_decl_after_decl.carbon

library "[[@TEST_NAME]]";

class C;
// CHECK:STDERR: fail_extern_decl_after_decl.carbon:[[@LINE+7]]:1: error: redeclaration of `class C` is redundant [RedeclRedundant]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_decl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_import_extern_decl_then_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_extern_decl_then_decl.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: decl.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_import_extern_decl_then_decl.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "extern_decl";
import library "decl";

// --- fail_import_decl_then_extern_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_decl_then_extern_decl.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_import_decl_then_extern_decl.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: decl.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
import library "decl";
import library "extern_decl";

// --- fail_import_extern_decl_then_def.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_extern_decl_then_def.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: def.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_import_extern_decl_then_def.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "extern_decl";
import library "def";

// --- fail_import_ownership_conflict.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: decl.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: def.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "extern_decl";
import library "decl";
import library "def";

// --- fail_todo_import_extern_decl_copy.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_import_extern_decl_copy.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: extern_decl_copy.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_import_extern_decl_copy.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "extern_decl";
import library "extern_decl_copy";

// --- fail_extern_decl_after_import_extern_decl.carbon

library "[[@TEST_NAME]]";

import library "extern_decl";

// CHECK:STDERR: fail_extern_decl_after_import_extern_decl.carbon:[[@LINE+8]]:1: error: redeclaration of `class C` is redundant [RedeclRedundant]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_import_extern_decl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: extern_decl.carbon:5:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_decl_after_import_extern_decl.carbon

library "[[@TEST_NAME]]";

import library "decl";

// CHECK:STDERR: fail_decl_after_import_extern_decl.carbon:[[@LINE+8]]:1: error: redeclarations of `class C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_after_import_extern_decl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: decl.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_def_after_import_extern_decl.carbon

library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_def_after_import_extern_decl.carbon:[[@LINE+8]]:1: error: redeclarations of `class C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_def_after_import_extern_decl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: def.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_extern_decl_after_import_def.carbon

library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_extern_decl_after_import_def.carbon:[[@LINE+8]]:1: error: redeclarations of `class C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_import_def.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: def.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
extern class C;

// CHECK:STDOUT: --- extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:

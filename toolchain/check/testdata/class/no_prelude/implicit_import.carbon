// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/implicit_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/implicit_import.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class C;

// --- basic.impl.carbon

impl library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C {}
//@dump-sem-ir-end

// --- redecl_after_def.carbon

library "[[@TEST_NAME]]";

class C {}

// --- fail_redecl_after_def.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_redecl_after_def.impl.carbon:[[@LINE+8]]:1: error: redeclaration of `class C` is redundant [RedeclRedundant]
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl_after_def.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: redecl_after_def.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
class C;

// --- redef_after_def.carbon

library "[[@TEST_NAME]]";

class C {}

// --- fail_redef_after_def.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_redef_after_def.impl.carbon:[[@LINE+8]]:1: error: redefinition of `class C` [RedeclRedef]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_redef_after_def.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: redef_after_def.carbon:4:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
class C {}

// --- def_alias.carbon

library "[[@TEST_NAME]]";

class C;
alias B = C;

// --- fail_def_alias.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_def_alias.impl.carbon:[[@LINE+8]]:7: error: duplicate name `B` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class B {}
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_def_alias.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: def_alias.carbon:5:7: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: alias B = C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
class B {}

// CHECK:STDOUT: --- basic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
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

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- decl.carbon

library "decl" api;

class C;

// --- extern_decl.carbon

library "extern_decl" api;

extern class C;

// --- extern_decl_copy.carbon

library "extern_decl_copy" api;

extern class C;

// --- def.carbon

library "def" api;

class C {}

// ============================================================================
// Test files
// ============================================================================

// --- fail_decl_fn_in_extern.carbon

library "fail_decl_fn_in_extern" api;

extern class C;
// CHECK:STDERR: fail_decl_fn_in_extern.carbon:[[@LINE+7]]:4: ERROR: Cannot declare a member of incomplete class `C`.
// CHECK:STDERR: fn C.F();
// CHECK:STDERR:    ^
// CHECK:STDERR: fail_decl_fn_in_extern.carbon:[[@LINE-4]]:1: Class was forward declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.F();

// --- fail_extern_def.carbon

library "fail_extern_def" api;

// CHECK:STDERR: fail_extern_def.carbon:[[@LINE+4]]:1: ERROR: `extern` not allowed on `class` declaration that provides a definition.
// CHECK:STDERR: extern class C {}
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extern class C {}

// --- fail_extern_decl_after_extern_decl.carbon

library "fail_extern_decl_after_extern_decl" api;

extern class C;
// CHECK:STDERR: fail_extern_decl_after_extern_decl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `class C` is redundant.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_extern_decl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_decl_after_extern_decl.carbon

library "fail_decl_after_extern_decl" api;

extern class C;
// CHECK:STDERR: fail_decl_after_extern_decl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `class C` is redundant.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_after_extern_decl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
class C;

// --- fail_extern_member_class.carbon

library "fail_extern_member_class" api;

class C {
  // CHECK:STDERR: fail_extern_member_class.carbon:[[@LINE+4]]:3: ERROR: `extern` not allowed on `class` declaration that is a member.
  // CHECK:STDERR:   extern class D;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern class D;
}

// --- todo_fail_def_after_extern_decl.carbon

library "fail_def_after_extern_decl" api;

extern class C;
class C {}

// --- fail_extern_decl_after_decl.carbon

library "fail_extern_decl_after_decl" api;

class C;
// CHECK:STDERR: fail_extern_decl_after_decl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `class C` is redundant.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_decl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
extern class C;

// --- fail_todo_import_extern_decl_then_decl.carbon

library "import_extern_decl_then_decl" api;

import library "extern_decl";
// CHECK:STDERR: fail_todo_import_extern_decl_then_decl.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: decl.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_todo_import_extern_decl_then_decl.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "decl";

// --- fail_todo_import_decl_then_extern_decl.carbon

library "import_decl_then_extern_decl" api;

import library "decl";
// CHECK:STDERR: fail_todo_import_decl_then_extern_decl.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_import_decl_then_extern_decl.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
import library "extern_decl";

// --- fail_todo_import_extern_decl_then_def.carbon

library "import_extern_decl_then_def" api;

import library "extern_decl";
// CHECK:STDERR: fail_todo_import_extern_decl_then_def.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "def";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: def.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_import_extern_decl_then_def.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "def";

// --- fail_import_ownership_conflict.carbon

library "fail_import_ownership_conflict" api;

import library "extern_decl";
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: decl.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "decl";
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "def";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: def.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_import_ownership_conflict.carbon:[[@LINE-21]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "def";

// --- fail_todo_import_extern_decl_copy.carbon

library "import_extern_decl_copy" api;

import library "extern_decl";
// CHECK:STDERR: fail_todo_import_extern_decl_copy.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "extern_decl_copy";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl_copy.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_import_extern_decl_copy.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_decl";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_decl.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
import library "extern_decl_copy";

// --- extern_decl_after_import_extern_decl.carbon

library "extern_decl_after_import_extern_decl" api;

import library "extern_decl";

extern class C;

// --- decl_after_import_extern_decl.carbon

library "decl_after_import_extern_decl" api;

import library "decl";

extern class C;

// --- def_after_import_extern_decl.carbon

library "def_after_import_extern_decl" api;

import library "def";

extern class C;

// --- extern_decl_after_import_def.carbon

library "extern_decl_after_import_def" api;

import library "def";

extern class C;

// CHECK:STDOUT: --- decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_decl_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_fn_in_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_decl_after_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_after_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_member_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .D = %D.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D;
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_def_after_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc5: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_decl_after_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extern_decl_then_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_decl_then_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extern_decl_then_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_ownership_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir3, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_extern_decl_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_decl_after_import_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_after_import_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- def_after_import_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_decl_after_import_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

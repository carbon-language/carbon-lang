// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/merging_with_indirections.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/merging_with_indirections.carbon

// --- a.carbon

package Other library "[[@TEST_NAME]]";

namespace NS1;
class NS1.A {}

// --- b.carbon

package Other library "[[@TEST_NAME]]";
import library "a";

namespace NS1;
class NS1.B {}

fn F() -> NS1.A { return {}; }

// --- main.carbon

import Other library "b";
import Other library "a";

fn Run() {
  // Note `Other.NS.A` is part of the return type here.
  Other.F();

  // Use `Other.NS.A` directly.
  var a: Other.NS1.A;

  // Ensure the type is equivalent.
  a = Other.F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS1 = %NS1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.NS1: <namespace> = import_ref Other//a, NS1, loaded
// CHECK:STDOUT:   %NS1: <namespace> = namespace %Other.NS1, [concrete] {
// CHECK:STDOUT:     .A = %Other.A
// CHECK:STDOUT:     .B = file.%B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.A: type = import_ref Other//a, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import_ref.8f2: <witness> = import_ref Other//a, loc5_14, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.ca8 = import_ref Other//a, inst18 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NS1 = imports.%NS1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %NS1: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Other.A
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %A = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %A = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NS1.ref: <namespace> = name_ref NS1, imports.%NS1 [concrete = imports.%NS1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Other.A [concrete = constants.%A]
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.ca8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param_patt: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_27.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_27.2: init %A = class_init (), %return [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc8_28: init %A = converted %.loc8_27.1, %.loc8_27.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   return %.loc8_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     .NS1 = %NS1
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//b, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Other.import_ref.8db: <witness> = import_ref Other//b, inst29 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.bbd = import_ref Other//b, inst30 [indirect], unloaded
// CHECK:STDOUT:   %Other.NS1: <namespace> = import_ref Other//b, NS1, loaded
// CHECK:STDOUT:   %NS1: <namespace> = namespace %Other.NS1, [concrete] {
// CHECK:STDOUT:     .A = %Other.A
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.A: type = import_ref Other//a, A, loaded [concrete = constants.%A]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A [from "b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.bbd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc7: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_11.1: ref %A = temporary_storage
// CHECK:STDOUT:   %F.call.loc7: init %A = call %F.ref.loc7() to %.loc7_11.1
// CHECK:STDOUT:   %.loc7_11.2: ref %A = temporary %.loc7_11.1, %F.call.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %.loc10_3: %A = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %.loc10_19: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Other.ref.loc10: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %NS1.ref: <namespace> = name_ref NS1, imports.%NS1 [concrete = imports.%NS1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Other.A [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %A = name_ref a, %a
// CHECK:STDOUT:   %Other.ref.loc13: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref.loc13: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13: ref %A = splice_block %a.ref {}
// CHECK:STDOUT:   %F.call.loc13: init %A = call %F.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign %a.ref, %F.call.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %A [from "b.carbon"];
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS1 = %NS1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS1: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct: %A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref Other//a, inst+3, loaded
// CHECK:STDOUT:   %NS1: <namespace> = namespace %import_ref.1, [template] {
// CHECK:STDOUT:     .A = %import_ref.2
// CHECK:STDOUT:     .B = file.%B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: type = import_ref Other//a, inst+4, loaded [template = constants.%A]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//a, inst+5, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS1 = imports.%NS1
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %NS1: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.2
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %A = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %A = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NS1.ref: <namespace> = name_ref NS1, imports.%NS1 [template = imports.%NS1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%import_ref.2 [template = constants.%A]
// CHECK:STDOUT:     %return.param: ref %A = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_27.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc8_27.2: init %A = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_28: init %A = converted %.loc8_27.1, %.loc8_27.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc8_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     .NS1 = %NS1
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %F.type = import_ref Other//b, inst+22, loaded [template = constants.%F]
// CHECK:STDOUT:   %import_ref.2 = import_ref Other//b, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.3: <namespace> = import_ref Other//b, inst+4, loaded
// CHECK:STDOUT:   %NS1: <namespace> = namespace %import_ref.3, [template] {
// CHECK:STDOUT:     .A = %import_ref.4
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: type = import_ref Other//a, inst+4, loaded [template = constants.%A]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Other.ref.loc7: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref.loc7: %F.type = name_ref F, imports.%import_ref.1 [template = constants.%F]
// CHECK:STDOUT:   %.loc7_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %F.call.loc7: init %A = call %F.ref.loc7() to %.loc7_10.1
// CHECK:STDOUT:   %.loc7_10.2: ref %A = temporary %.loc7_10.1, %F.call.loc7
// CHECK:STDOUT:   %Other.ref.loc10: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %NS1.ref: <namespace> = name_ref NS1, imports.%NS1 [template = imports.%NS1]
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%import_ref.4 [template = constants.%A]
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %A = name_ref a, %a
// CHECK:STDOUT:   %Other.ref.loc13: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %F.ref.loc13: %F.type = name_ref F, imports.%import_ref.1 [template = constants.%F]
// CHECK:STDOUT:   %.loc13: ref %A = splice_block %a.ref {}
// CHECK:STDOUT:   %F.call.loc13: init %A = call %F.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign %a.ref, %F.call.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %A;
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
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
// CHECK:STDOUT:   %Other.import_ref.ca8 = import_ref Other//a, inst19 [no loc], unloaded
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
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NS1.ref: <namespace> = name_ref NS1, imports.%NS1 [concrete = imports.%NS1]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Other.A [concrete = constants.%A]
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: fn @F() -> %return.param: %A {
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.272: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.d74: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%A) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.478: %T.as.Destroy.impl.Op.type.d74 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.07e: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.478, @T.as.Destroy.impl.Op(%A) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .F = %Other.F
// CHECK:STDOUT:     .NS1 = %NS1.b9a
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.F: %F.type = import_ref Other//b, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Other.import_ref.8db: <witness> = import_ref Other//b, inst31 [indirect], loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Other.import_ref.bbd = import_ref Other//b, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Other.NS1: <namespace> = import_ref Other//b, NS1, loaded
// CHECK:STDOUT:   %NS1.b9a: <namespace> = namespace %Other.NS1, [concrete] {
// CHECK:STDOUT:     .A = %Other.A
// CHECK:STDOUT:     import Other//b
// CHECK:STDOUT:     import Other//a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.A: type = import_ref Other//a, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.272 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.272 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %.loc10: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Other.ref.loc10: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %NS1.ref: <namespace> = name_ref NS1, imports.%NS1.b9a [concrete = imports.%NS1.b9a]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%Other.A [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %A = name_ref a, %a
// CHECK:STDOUT:   %Other.ref.loc13: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %F.ref.loc13: %F.type = name_ref F, imports.%Other.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13: ref %A = splice_block %a.ref {}
// CHECK:STDOUT:   %F.call.loc13: init %A = call %F.ref.loc13() to %.loc13
// CHECK:STDOUT:   assign %a.ref, %F.call.loc13
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc13: <bound method> = bound_method %.loc13, constants.%T.as.Destroy.impl.Op.478
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.478, @T.as.Destroy.impl.Op(constants.%A) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %.loc13, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13: %ptr.07e = addr_of %.loc13
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc13: init %empty_tuple.type = call %bound_method.loc13(%addr.loc13)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %a.var, constants.%T.as.Destroy.impl.Op.478
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.478, @T.as.Destroy.impl.Op(constants.%A) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %a.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10: %ptr.07e = addr_of %a.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc7: <bound method> = bound_method %.loc7_11.1, constants.%T.as.Destroy.impl.Op.478
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.478, @T.as.Destroy.impl.Op(constants.%A) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %.loc7_11.1, %T.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc7: %ptr.07e = addr_of %.loc7_11.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc7: init %empty_tuple.type = call %bound_method.loc7(%addr.loc7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "b.carbon"];
// CHECK:STDOUT:

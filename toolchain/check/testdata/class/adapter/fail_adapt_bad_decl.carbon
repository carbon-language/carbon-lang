// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_decl.carbon

// --- fail_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  adapt 100;
}

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:18: error: member name `F` not found in `Bad` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn Use(b: Bad) { b.F(); }
// CHECK:STDERR:                  ^~~
// CHECK:STDERR:
fn Use(b: Bad) { b.F(); }

// --- fail_extend_not_type.carbon

library "[[@TEST_NAME]]";

class Bad {
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   extend adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_extend_not_type.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   extend adapt 100;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend adapt 100;
}

// No diagnostic here, we don't know what names Bad has.
fn Use(b: Bad) { b.F(); }

// --- fail_repeated.carbon

library "[[@TEST_NAME]]";

class MultipleAdapts {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class MultipleAdaptsSameType {
  adapt ();
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE+7]]:3: error: multiple `adapt` declarations in class [AdaptDeclRepeated]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_repeated.carbon:[[@LINE-4]]:3: note: previous `adapt` declaration is here [ClassSpecificDeclPrevious]
  // CHECK:STDERR:   adapt ();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt ();
}

// --- fail_bad_scope.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:1: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
// CHECK:STDERR: adapt {};
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
adapt {};

interface I {
  // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:3: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
  // CHECK:STDERR:   adapt {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  adapt {};
}

class C {
  interface I {
    // CHECK:STDERR: fail_bad_scope.carbon:[[@LINE+4]]:5: error: `adapt` declaration outside class [ClassSpecificDeclOutsideClass]
    // CHECK:STDERR:     adapt {};
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    adapt {};
  }
}

// CHECK:STDOUT: --- fail_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bad: type = class_type @Bad [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Bad = %Bad.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Bad.decl: type = class_decl @Bad [concrete = constants.%Bad] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %b.patt: %Bad = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %Bad = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Bad = value_param call_param0
// CHECK:STDOUT:     %Bad.ref: type = name_ref Bad, file.%Bad.decl [concrete = constants.%Bad]
// CHECK:STDOUT:     %b: %Bad = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bad {
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %.loc12: type = converted %int_100, <error> [concrete = <error>]
// CHECK:STDOUT:   adapt_decl <error> [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bad
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use(%b.param_patt: %Bad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %Bad = name_ref b, %b
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extend_not_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bad: type = class_type @Bad [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Bad = %Bad.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Bad.decl: type = class_decl @Bad [concrete = constants.%Bad] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %b.patt: %Bad = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %Bad = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Bad = value_param call_param0
// CHECK:STDOUT:     %Bad.ref: type = name_ref Bad, file.%Bad.decl [concrete = constants.%Bad]
// CHECK:STDOUT:     %b: %Bad = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Bad {
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %.loc12: type = converted %int_100, <error> [concrete = <error>]
// CHECK:STDOUT:   adapt_decl <error> [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Bad
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use(%b.param_patt: %Bad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: %Bad = name_ref b, %b
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MultipleAdapts: type = class_type @MultipleAdapts [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %MultipleAdaptsSameType: type = class_type @MultipleAdaptsSameType [concrete]
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
// CHECK:STDOUT:     .MultipleAdapts = %MultipleAdapts.decl
// CHECK:STDOUT:     .MultipleAdaptsSameType = %MultipleAdaptsSameType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MultipleAdapts.decl: type = class_decl @MultipleAdapts [concrete = constants.%MultipleAdapts] {} {}
// CHECK:STDOUT:   %MultipleAdaptsSameType.decl: type = class_decl @MultipleAdaptsSameType [concrete = constants.%MultipleAdaptsSameType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MultipleAdapts {
// CHECK:STDOUT:   %.loc5_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11: type = converted %.loc5_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc5_11 [concrete]
// CHECK:STDOUT:   %.loc13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MultipleAdapts
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @MultipleAdaptsSameType {
// CHECK:STDOUT:   %.loc17_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_11: type = converted %.loc17_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc17_11 [concrete]
// CHECK:STDOUT:   %.loc25: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%MultipleAdaptsSameType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.1> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type.e30: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT:   %Self.6ef: %I.type.e30 = bind_symbolic_name Self, 0 [symbolic]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc8: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %I.decl: type = interface_decl @I.1 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.1 {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: %I.type.e30 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.6ef]
// CHECK:STDOUT:   %.loc24: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: type = interface_decl @I.2 [concrete = constants.%I.type.e30] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:

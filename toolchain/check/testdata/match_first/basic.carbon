// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ^ARGS: -v compile %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/match_first/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/match_first/basic.carbon

// --- empty.carbon
library "[[@TEST_NAME]]";

match_first {}

// --- todo_fail_class_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  class C;
}

// --- todo_fail_interface_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  interface Z;
}

// --- todo_fail_constraint_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  constraint N;
}

// --- todo_fail_function_inside.carbon
library "[[@TEST_NAME]]";

match_first {
  // TODO: Only impls are allowed inside match_first.
  fn F();
}

// --- fail_final_inside.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  // CHECK:STDERR: fail_final_inside.carbon:[[@LINE+3]]:3-22: error: `final impl` in `match_first` block [FinalImplInMatchFirst]
  // CHECK:STDERR: fail_final_inside.carbon:[[@LINE-2]]:1-13: note: the `match_first` block can be modified as `final` instead [FinalImplInMatchFirstNote]
  // CHECK:STDERR:
  final impl () as Z {}
}

// --- fail_final_redecl.carbon
library "[[@TEST_NAME]]";

interface Z {}

// CHECK:STDERR: fail_final_redecl.carbon:[[@LINE+2]]:1-19: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
final impl () as Z;
match_first {
  // This is not seen as a redecl of the `final impl`. It's not valid to write
  // an `impl` in `match_first` that overlaps a `final impl`, but we already
  // have a diagnostic.
  impl () as Z {}
}

// --- fail_final_redecl_defined_once.carbon
library "[[@TEST_NAME]]";

interface Z {}

final impl () as Z {}
match_first {
  // This is not seen as a redecl of the `final impl`. It's not valid to write
  // an `impl` in `match_first` that overlaps a `final impl`, but we already
  // have a diagnostic.
  // CHECK:STDERR: fail_final_redecl_defined_once.carbon:[[@LINE+2]]:3-15: error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:
  impl () as Z;
}

// --- fail_final_redecl_defined_twice.carbon
library "[[@TEST_NAME]]";

interface Z {}

final impl () as Z {}
match_first {
  // This is not seen as a redecl of the `final impl`. It's not valid to write
  // an `impl` in `match_first` that overlaps a `final impl`, but we already
  // have a diagnostic.
  // CHECK:STDERR: fail_final_redecl_defined_twice.carbon:[[@LINE+3]]:3-16: error: `impl` will never be used [ImplFinalOverlapsNonFinal]
  // CHECK:STDERR: fail_final_redecl_defined_twice.carbon:[[@LINE-6]]:1-20: note: `final impl` declared here would always be used instead [ImplFinalOverlapsNonFinalNote]
  // CHECK:STDERR:
  impl () as Z {}
}

// --- fail_final_final_redecl.carbon
library "[[@TEST_NAME]]";

interface Z {}

final impl () as Z;
match_first {
  // This _is_ seen as a redecl of the `final impl`. It's not valid to write
  // an `impl` in `match_first` that overlaps a `final impl`, but we already
  // have a diagnostic.
  // CHECK:STDERR: fail_final_final_redecl.carbon:[[@LINE+3]]:3-22: error: `final impl` in `match_first` block [FinalImplInMatchFirst]
  // CHECK:STDERR: fail_final_final_redecl.carbon:[[@LINE-5]]:1-13: note: the `match_first` block can be modified as `final` instead [FinalImplInMatchFirstNote]
  // CHECK:STDERR:
  final impl () as Z {}
}

// --- decl.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
}
//@dump-sem-ir-begin
impl () as Z {}
//@dump-sem-ir-end

// --- definition.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  //@dump-sem-ir-begin
  impl () as Z {}
  //@dump-sem-ir-end
}

// --- redecl_defined.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z;
match_first {
  //@dump-sem-ir-begin
  impl () as Z {}
  //@dump-sem-ir-end
}

// --- redecl.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z;
match_first {
  impl () as Z;
}
impl () as Z {}

// --- redecl_after_defined.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z {}
match_first {
  impl () as Z;
}

// --- fail_redecl_with_definition_after_defined.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z {}
match_first {
  // CHECK:STDERR: fail_redecl_with_definition_after_defined.carbon:[[@LINE+3]]:3-16: error: redefinition of `impl () as Z` [ImplRedefinition]
  // CHECK:STDERR: fail_redecl_with_definition_after_defined.carbon:[[@LINE-3]]:1-14: note: previous definition was here [ImplPreviousDefinition]
  // CHECK:STDERR:
  impl () as Z {}
}
// --- inside_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  match_first {
    impl Self as Z {}
  }
}

// --- inside_function.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  match_first {
    impl () as Z {}
  }
}

// --- fail_inside_interface.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface Y {
  // CHECK:STDERR: fail_inside_interface.carbon:[[@LINE+3]]:3-15: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR: fail_inside_interface.carbon:[[@LINE-2]]:1-13: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR:
  match_first {
    impl Self as Z {}
  }
}

// --- fail_inside_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint N {
  // CHECK:STDERR: fail_inside_constraint.carbon:[[@LINE+3]]:3-15: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR: fail_inside_constraint.carbon:[[@LINE-2]]:1-14: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR:
  match_first {
    impl Self as Z {}
  }
}

// --- fail_inside_match_first.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  // CHECK:STDERR: fail_inside_match_first.carbon:[[@LINE+3]]:3-15: error: found `match_first` in invalid scope; expected namespace or class [MatchFirstInWrongScope]
  // CHECK:STDERR: fail_inside_match_first.carbon: note: in enclosing scope here [MatchFirstInWrongScopeNote]
  // CHECK:STDERR:
  match_first {
    impl () as Z {}
  }
}

// --- fail_redecl_in_same_block.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
  // CHECK:STDERR: fail_redecl_in_same_block.carbon:[[@LINE+3]]:3-15: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_redecl_in_same_block.carbon:[[@LINE-2]]:3-15: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z;
}
impl () as Z {}

// --- fail_defined_in_same_block.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
  // CHECK:STDERR: fail_defined_in_same_block.carbon:[[@LINE+3]]:3-16: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_defined_in_same_block.carbon:[[@LINE-2]]:3-15: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z {}
}

// --- fail_redecl_after_defined_in_same_block.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z {}
  // CHECK:STDERR: fail_redecl_after_defined_in_same_block.carbon:[[@LINE+3]]:3-15: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_redecl_after_defined_in_same_block.carbon:[[@LINE-2]]:3-16: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z;
}

// --- fail_two_redecl_in_same_block.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z {}
match_first {
  impl () as Z;
  // TODO: This "previous declaration" location is wrong. The redecl came after.
  // CHECK:STDERR: fail_two_redecl_in_same_block.carbon:[[@LINE+3]]:3-15: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_two_redecl_in_same_block.carbon:[[@LINE-3]]:3-15: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z;
}

// --- fail_redecl_in_different_block.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
}
match_first {
  // CHECK:STDERR: fail_redecl_in_different_block.carbon:[[@LINE+3]]:3-15: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_redecl_in_different_block.carbon:[[@LINE-4]]:3-15: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z;
}
impl () as Z {}

// --- fail_redecl_defined_in_different_block_firset.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z {}
}
match_first {
  // CHECK:STDERR: fail_redecl_defined_in_different_block_firset.carbon:[[@LINE+3]]:3-15: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_redecl_defined_in_different_block_firset.carbon:[[@LINE-4]]:3-16: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z;
}

// --- fail_redecl_defined_in_different_block_second.carbon
library "[[@TEST_NAME]]";

interface Z {}

match_first {
  impl () as Z;
}
match_first {
  // CHECK:STDERR: fail_redecl_defined_in_different_block_second.carbon:[[@LINE+3]]:3-16: error: impl declared in `match_first` more than once [ImplInTwoMatchFirst]
  // CHECK:STDERR: fail_redecl_defined_in_different_block_second.carbon:[[@LINE-4]]:3-15: note: previous declaration here [ImplInTwoMatchFirstNote]
  // CHECK:STDERR:
  impl () as Z {}
}

// --- final_block_redecl.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z;
final match_first {
  impl () as Z;
}
//@dump-sem-ir-begin
impl () as Z {}
//@dump-sem-ir-end

// --- final_block_decl.carbon
library "[[@TEST_NAME]]";

interface Z {}

final match_first {
  impl () as Z;
}
//@dump-sem-ir-begin
impl () as Z {}
//@dump-sem-ir-end

// --- final_block_defined.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl () as Z;
final match_first {
  //@dump-sem-ir-begin
  impl () as Z {}
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref.loc9: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc9_7.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc6_9.2 as %Z.ref.loc6 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref.loc6
// CHECK:STDOUT:   match_first = position 0
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @empty_tuple.type.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_9.2: type = converted %.loc7_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc7_9.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc7_9.2 as %Z.ref {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (), @empty_tuple.type.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   match_first = position 0
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_defined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_9.2: type = converted %.loc8_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref.loc8: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc8_9.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc5_7.2 as %Z.ref.loc5 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref.loc5
// CHECK:STDOUT:   match_first = position 0
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_block_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref.loc10: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc10_7.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc5_7.2 as %Z.ref.loc5 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref.loc5
// CHECK:STDOUT:   match_first = position 0, final
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_block_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref.loc9: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc9_7.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc6_9.2 as %Z.ref.loc6 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref.loc6
// CHECK:STDOUT:   match_first = position 0, final
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_block_defined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.78e: <witness> = impl_self_witness %empty_tuple.type, @Z [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_9.2: type = converted %.loc8_9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Z.ref.loc8: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @empty_tuple.type.as.Z.impl.%.loc8_9.2, @Z [concrete = constants.%.78e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.Z.impl: %.loc5_7.2 as %Z.ref.loc5 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref.loc5
// CHECK:STDOUT:   match_first = position 0, final
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:

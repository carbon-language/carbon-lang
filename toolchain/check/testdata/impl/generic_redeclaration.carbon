// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/generic_redeclaration.carbon

// --- fail_todo_same_self_and_interface.carbon

library "[[@TEST_NAME]]";

interface Interface {}

interface I {}
interface J {}
interface K {}
interface L {}

// TODO: Put these in a match_first so the test will pass again.
impl forall [T:! I] T as Interface;
impl forall [T:! J] T as Interface;
impl forall [T:! K] T as Interface;
impl forall [T:! L] T as Interface;

// These are different impls, so they are not redefinitions, even though the
// self type and constraint type are the same.
impl forall [T:! I] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T:! J] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-4]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T:! I] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! J] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T:! K] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-12]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T:! I] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! K] T as Interface {}
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl forall [T:! L] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_same_self_and_interface.carbon:[[@LINE-20]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl forall [T:! I] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! L] T as Interface {}

// --- fail_same_self_and_interface_redefined.carbon

library "[[@TEST_NAME]]";

interface I {}
interface J {}

impl forall [T:! I] T as J {}
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `impl T as J` [ImplRedefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE-4]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! I] T as J {}

// --- fail_same_type_different_spelling.carbon

library "[[@TEST_NAME]]";

class C;
interface I {}

// Since the spelling is different, it's not caught as a redefinition, but it
// is diagnosed as an overlapping impl without using a `match_first` block.
impl C as I {}
// CHECK:STDERR: fail_same_type_different_spelling.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl (C, C).0 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_same_type_different_spelling.carbon:[[@LINE-4]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl (C, C).0 as I {}

// --- fail_redefinition_generic_regions.carbon

interface I {}

impl forall [T:! type] T as I {
  fn A() {}
}

// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE+7]]:1: error: redefinition of `impl T as I` [ImplRedefinition]
// CHECK:STDERR: impl forall [T:! type] T as I {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE-7]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl forall [T:! type] T as I {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as I {
  // Although not referenced, B has the same generic region index of A, and
  // makes C a different index. We used to merge, and this was a crash.
  fn B() {}
  fn C() -> () { return (); }
  fn D() -> () {
    return C();
  }
}

// CHECK:STDOUT: --- fail_todo_same_self_and_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.46e: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.bf6: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.9b6: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.c04: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.7ee: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.1b7: type = symbolic_binding_type T, 0, %T.7ee [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.9c5: <witness> = impl_witness file.%Interface.impl_witness_table.loc12, @T.binding.as_type.as.Interface.impl.c91(%T.7ee) [symbolic]
// CHECK:STDOUT:   %T.bf6: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.84b: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.09d: type = symbolic_binding_type T, 0, %T.bf6 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.df0: <witness> = impl_witness file.%Interface.impl_witness_table.loc13, @T.binding.as_type.as.Interface.impl.117(%T.bf6) [symbolic]
// CHECK:STDOUT:   %T.9b6: %K.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.0fc: type = pattern_type %K.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.4c4: type = symbolic_binding_type T, 0, %T.9b6 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.39f: <witness> = impl_witness file.%Interface.impl_witness_table.loc14, @T.binding.as_type.as.Interface.impl.96a(%T.9b6) [symbolic]
// CHECK:STDOUT:   %T.c04: %L.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ebc: type = pattern_type %L.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.19c: type = symbolic_binding_type T, 0, %T.c04 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.81c: <witness> = impl_witness file.%Interface.impl_witness_table.loc15, @T.binding.as_type.as.Interface.impl.9a4(%T.c04) [symbolic]
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
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.c91 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.09a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12: %I.type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.7ee)]
// CHECK:STDOUT:     %T.as_type.loc12: type = facet_access_type %T.ref.loc12 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.1b7)]
// CHECK:STDOUT:     %.loc12_21: type = converted %T.ref.loc12, %T.as_type.loc12 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.1b7)]
// CHECK:STDOUT:     %Interface.ref.loc12: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc12_18: type = splice_block %I.ref.loc12 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc12: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.7ee)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.impl_witness_table.loc12 = impl_witness_table (), @T.binding.as_type.as.Interface.impl.c91 [concrete]
// CHECK:STDOUT:   %Interface.impl_witness.loc12: <witness> = impl_witness %Interface.impl_witness_table.loc12, @T.binding.as_type.as.Interface.impl.c91(constants.%T.7ee) [symbolic = @T.binding.as_type.as.Interface.impl.c91.%Interface.impl_witness (constants.%Interface.impl_witness.9c5)]
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.117 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.84b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13: %J.type = name_ref T, %T.loc13_14.1 [symbolic = %T.loc13_14.2 (constants.%T.bf6)]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.09d)]
// CHECK:STDOUT:     %.loc13_21: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.09d)]
// CHECK:STDOUT:     %Interface.ref.loc13: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc13_18: type = splice_block %J.ref.loc13 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc13: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_14.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.bf6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.impl_witness_table.loc13 = impl_witness_table (), @T.binding.as_type.as.Interface.impl.117 [concrete]
// CHECK:STDOUT:   %Interface.impl_witness.loc13: <witness> = impl_witness %Interface.impl_witness_table.loc13, @T.binding.as_type.as.Interface.impl.117(constants.%T.bf6) [symbolic = @T.binding.as_type.as.Interface.impl.117.%Interface.impl_witness (constants.%Interface.impl_witness.df0)]
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.96a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.0fc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc14: %K.type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T.9b6)]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.4c4)]
// CHECK:STDOUT:     %.loc14_21: type = converted %T.ref.loc14, %T.as_type.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.4c4)]
// CHECK:STDOUT:     %Interface.ref.loc14: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc14_18: type = splice_block %K.ref.loc14 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref.loc14: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_14.1: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.9b6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.impl_witness_table.loc14 = impl_witness_table (), @T.binding.as_type.as.Interface.impl.96a [concrete]
// CHECK:STDOUT:   %Interface.impl_witness.loc14: <witness> = impl_witness %Interface.impl_witness_table.loc14, @T.binding.as_type.as.Interface.impl.96a(constants.%T.9b6) [symbolic = @T.binding.as_type.as.Interface.impl.96a.%Interface.impl_witness (constants.%Interface.impl_witness.39f)]
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.9a4 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ebc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc15: %L.type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T.c04)]
// CHECK:STDOUT:     %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.19c)]
// CHECK:STDOUT:     %.loc15_21: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.19c)]
// CHECK:STDOUT:     %Interface.ref.loc15: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc15_18: type = splice_block %L.ref.loc15 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref.loc15: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_14.1: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T.c04)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.impl_witness_table.loc15 = impl_witness_table (), @T.binding.as_type.as.Interface.impl.9a4 [concrete]
// CHECK:STDOUT:   %Interface.impl_witness.loc15: <witness> = impl_witness %Interface.impl_witness_table.loc15, @T.binding.as_type.as.Interface.impl.9a4(constants.%T.c04) [symbolic = @T.binding.as_type.as.Interface.impl.9a4.%Interface.impl_witness (constants.%Interface.impl_witness.81c)]
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.c91 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.09a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: %I.type = name_ref T, %T.loc19 [symbolic = %T.loc12_14.2 (constants.%T.7ee)]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.1b7)]
// CHECK:STDOUT:     %.loc19_21: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.1b7)]
// CHECK:STDOUT:     %Interface.ref.loc19: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc19_18: type = splice_block %I.ref.loc19 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc19: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.7ee)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.117 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.84b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc27: %J.type = name_ref T, %T.loc27 [symbolic = %T.loc13_14.2 (constants.%T.bf6)]
// CHECK:STDOUT:     %T.as_type.loc27: type = facet_access_type %T.ref.loc27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.09d)]
// CHECK:STDOUT:     %.loc27_21: type = converted %T.ref.loc27, %T.as_type.loc27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.09d)]
// CHECK:STDOUT:     %Interface.ref.loc27: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc27_18: type = splice_block %J.ref.loc27 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc27: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.bf6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.96a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.0fc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc35: %K.type = name_ref T, %T.loc35 [symbolic = %T.loc14_14.2 (constants.%T.9b6)]
// CHECK:STDOUT:     %T.as_type.loc35: type = facet_access_type %T.ref.loc35 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.4c4)]
// CHECK:STDOUT:     %.loc35_21: type = converted %T.ref.loc35, %T.as_type.loc35 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.4c4)]
// CHECK:STDOUT:     %Interface.ref.loc35: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc35_18: type = splice_block %K.ref.loc35 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref.loc35: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc35: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.9b6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.9a4 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ebc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc43: %L.type = name_ref T, %T.loc43 [symbolic = %T.loc15_14.2 (constants.%T.c04)]
// CHECK:STDOUT:     %T.as_type.loc43: type = facet_access_type %T.ref.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.19c)]
// CHECK:STDOUT:     %.loc43_21: type = converted %T.ref.loc43, %T.as_type.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.19c)]
// CHECK:STDOUT:     %Interface.ref.loc43: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc43_18: type = splice_block %L.ref.loc43 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref.loc43: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc43: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T.c04)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bf6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.9b6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.c04]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.c91(%T.loc12_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc12_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.7ee)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.1b7)]
// CHECK:STDOUT:   %Interface.impl_witness: <witness> = impl_witness file.%Interface.impl_witness_table.loc12, @T.binding.as_type.as.Interface.impl.c91(%T.loc12_14.2) [symbolic = %Interface.impl_witness (constants.%Interface.impl_witness.9c5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12_21 as %Interface.ref.loc12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Interface.impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.117(%T.loc13_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc13_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.bf6)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.09d)]
// CHECK:STDOUT:   %Interface.impl_witness: <witness> = impl_witness file.%Interface.impl_witness_table.loc13, @T.binding.as_type.as.Interface.impl.117(%T.loc13_14.2) [symbolic = %Interface.impl_witness (constants.%Interface.impl_witness.df0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13_21 as %Interface.ref.loc13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Interface.impl_witness.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.96a(%T.loc14_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc14_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.9b6)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc14_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.4c4)]
// CHECK:STDOUT:   %Interface.impl_witness: <witness> = impl_witness file.%Interface.impl_witness_table.loc14, @T.binding.as_type.as.Interface.impl.96a(%T.loc14_14.2) [symbolic = %Interface.impl_witness (constants.%Interface.impl_witness.39f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_21 as %Interface.ref.loc14 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Interface.impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.9a4(%T.loc15_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc15_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T.c04)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc15_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.19c)]
// CHECK:STDOUT:   %Interface.impl_witness: <witness> = impl_witness file.%Interface.impl_witness_table.loc15, @T.binding.as_type.as.Interface.impl.9a4(%T.loc15_14.2) [symbolic = %Interface.impl_witness (constants.%Interface.impl_witness.81c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc15_21 as %Interface.ref.loc15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Interface.impl_witness.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.c91(constants.%T.7ee) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.7ee
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.1b7
// CHECK:STDOUT:   %Interface.impl_witness => constants.%Interface.impl_witness.9c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.117(constants.%T.bf6) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%T.bf6
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.09d
// CHECK:STDOUT:   %Interface.impl_witness => constants.%Interface.impl_witness.df0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.96a(constants.%T.9b6) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T.9b6
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.4c4
// CHECK:STDOUT:   %Interface.impl_witness => constants.%Interface.impl_witness.39f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.9a4(constants.%T.c04) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T.c04
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.19c
// CHECK:STDOUT:   %Interface.impl_witness => constants.%Interface.impl_witness.81c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.7ee: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.bf6: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table, @T.binding.as_type.as.J.impl.ed0bf9.1(%T) [symbolic]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.J.impl.ed0bf9.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc7_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc7_18: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.J.impl.ed0bf9.1 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table, @T.binding.as_type.as.J.impl.ed0bf9.1(constants.%T) [symbolic = @T.binding.as_type.as.J.impl.ed0bf9.1.%J.impl_witness (constants.%J.impl_witness)]
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.J.impl.ed0bf9.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc15_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc15_18: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.7ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bf6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.J.impl.ed0bf9.1(%T.loc7_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc7_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table, @T.binding.as_type.as.J.impl.ed0bf9.1(%T.loc7_14.2) [symbolic = %J.impl_witness (constants.%J.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7_21 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%J.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.J.impl.ed0bf9.2(%T.loc15_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc15_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc15_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc15_21 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.J.impl.ed0bf9.1(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.J.impl.ed0bf9.2(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.f6038c.1: <witness> = impl_witness file.%I.impl_witness_table.loc9 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %I.impl_witness.f6038c.2: <witness> = impl_witness file.%I.impl_witness_table.loc17 [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl.7de31e.1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc9 = impl_witness_table (), @C.as.I.impl.7de31e.1 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc9: <witness> = impl_witness %I.impl_witness_table.loc9 [concrete = constants.%I.impl_witness.f6038c.1]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.7de31e.2 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc17_7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc17_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc17_11.1: %tuple.type = tuple_literal (%C.ref.loc17_7, %C.ref.loc17_10)
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %tuple: %tuple.type = tuple_value (%C.ref.loc17_7, %C.ref.loc17_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc17_11.2: %tuple.type = converted %.loc17_11.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %tuple.elem0: type = tuple_access %.loc17_11.2, element0 [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc17 = impl_witness_table (), @C.as.I.impl.7de31e.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc17: <witness> = impl_witness %I.impl_witness_table.loc17 [concrete = constants.%I.impl_witness.f6038c.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.7de31e.1: %C.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.7de31e.2: %tuple.elem0 as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefinition_generic_regions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @T.as.I.impl.5103f4.1(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.5103f4.1(%T) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as.I.impl.A: %T.as.I.impl.A.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.5103f4.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B: %T.as.I.impl.B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.5103f4.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.C: %T.as.I.impl.C.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.5103f4.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D: %T.as.I.impl.D.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %T.as.I.impl.C.specific_fn: <specific function> = specific_function %T.as.I.impl.C, @T.as.I.impl.C(%T) [symbolic]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as.I.impl.5103f4.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_14.2 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @T.as.I.impl.5103f4.1 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl.5103f4.1(constants.%T) [symbolic = @T.as.I.impl.5103f4.1.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   impl_decl @T.as.I.impl.5103f4.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_14.2 [symbolic = %T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.5103f4.1(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @T.as.I.impl.5103f4.1(%T.loc4_14.1) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.5103f4.1(%T.loc4_14.1) [symbolic = %T.as.I.impl.A.type (constants.%T.as.I.impl.A.type)]
// CHECK:STDOUT:   %T.as.I.impl.A: @T.as.I.impl.5103f4.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = struct_value () [symbolic = %T.as.I.impl.A (constants.%T.as.I.impl.A)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.A.decl: @T.as.I.impl.5103f4.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = fn_decl @T.as.I.impl.A [symbolic = @T.as.I.impl.5103f4.1.%T.as.I.impl.A (constants.%T.as.I.impl.A)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .A = %T.as.I.impl.A.decl
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.5103f4.2(%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T.loc15_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.5103f4.2(%T.loc15_14.1) [symbolic = %T.as.I.impl.B.type (constants.%T.as.I.impl.B.type)]
// CHECK:STDOUT:   %T.as.I.impl.B: @T.as.I.impl.5103f4.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = struct_value () [symbolic = %T.as.I.impl.B (constants.%T.as.I.impl.B)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.5103f4.2(%T.loc15_14.1) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.5103f4.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.5103f4.2(%T.loc15_14.1) [symbolic = %T.as.I.impl.D.type (constants.%T.as.I.impl.D.type)]
// CHECK:STDOUT:   %T.as.I.impl.D: @T.as.I.impl.5103f4.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = struct_value () [symbolic = %T.as.I.impl.D (constants.%T.as.I.impl.D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.B.decl: @T.as.I.impl.5103f4.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = fn_decl @T.as.I.impl.B [symbolic = @T.as.I.impl.5103f4.2.%T.as.I.impl.B (constants.%T.as.I.impl.B)] {} {}
// CHECK:STDOUT:     %T.as.I.impl.C.decl: @T.as.I.impl.5103f4.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = fn_decl @T.as.I.impl.C [symbolic = @T.as.I.impl.5103f4.2.%T.as.I.impl.C (constants.%T.as.I.impl.C)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc19_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc19_14.2: type = converted %.loc19_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.as.I.impl.D.decl: @T.as.I.impl.5103f4.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = fn_decl @T.as.I.impl.D [symbolic = @T.as.I.impl.5103f4.2.%T.as.I.impl.D (constants.%T.as.I.impl.D)] {
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc20_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc20_14.2: type = converted %.loc20_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .B = %T.as.I.impl.B.decl
// CHECK:STDOUT:     .C = %T.as.I.impl.C.decl
// CHECK:STDOUT:     .D = %T.as.I.impl.D.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.A(@T.as.I.impl.5103f4.1.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.B(@T.as.I.impl.5103f4.2.%T.loc15_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.C(@T.as.I.impl.5103f4.2.%T.loc15_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_26.2: init %empty_tuple.type = tuple_init () to %return [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_27: init %empty_tuple.type = converted %.loc19_26.1, %.loc19_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc19_27 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.D(@T.as.I.impl.5103f4.2.%T.loc15_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.5103f4.2(%T) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.C.specific_fn.loc21_12.2: <specific function> = specific_function %T.as.I.impl.C, @T.as.I.impl.C(%T) [symbolic = %T.as.I.impl.C.specific_fn.loc21_12.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc21: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = specific_constant @T.as.I.impl.5103f4.2.%T.as.I.impl.C.decl, @T.as.I.impl.5103f4.2(constants.%T) [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %C.ref: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = name_ref C, %.loc21 [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %T.as.I.impl.C.specific_fn.loc21_12.1: <specific function> = specific_function %C.ref, @T.as.I.impl.C(constants.%T) [symbolic = %T.as.I.impl.C.specific_fn.loc21_12.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:     %T.as.I.impl.C.call: init %empty_tuple.type = call %T.as.I.impl.C.specific_fn.loc21_12.1()
// CHECK:STDOUT:     return %T.as.I.impl.C.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.5103f4.1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type => constants.%T.as.I.impl.A.type
// CHECK:STDOUT:   %T.as.I.impl.A => constants.%T.as.I.impl.A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.5103f4.2(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.B.type => constants.%T.as.I.impl.B.type
// CHECK:STDOUT:   %T.as.I.impl.B => constants.%T.as.I.impl.B
// CHECK:STDOUT:   %T.as.I.impl.C.type => constants.%T.as.I.impl.C.type
// CHECK:STDOUT:   %T.as.I.impl.C => constants.%T.as.I.impl.C
// CHECK:STDOUT:   %T.as.I.impl.D.type => constants.%T.as.I.impl.D.type
// CHECK:STDOUT:   %T.as.I.impl.D => constants.%T.as.I.impl.D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.B(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.D(constants.%T) {}
// CHECK:STDOUT:

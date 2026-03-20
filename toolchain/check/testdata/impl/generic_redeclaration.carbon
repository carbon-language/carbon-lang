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
// CHECK:STDOUT:   %Self.887: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.44d: %K.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.cbc: %L.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.651: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.3af: type = symbolic_binding_type T, 0, %T.651 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.779: <witness> = impl_witness @T.binding.as_type.as.Interface.impl.973.%Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.973(%T.651) [symbolic]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.612: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.50b: type = symbolic_binding_type T, 0, %T.612 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.8e2: <witness> = impl_witness @T.binding.as_type.as.Interface.impl.5eb.%Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.5eb(%T.612) [symbolic]
// CHECK:STDOUT:   %pattern_type.250: type = pattern_type %K.type [concrete]
// CHECK:STDOUT:   %T.9b2: %K.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.bec: type = symbolic_binding_type T, 0, %T.9b2 [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.84e: <witness> = impl_witness @T.binding.as_type.as.Interface.impl.edf.%Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.edf(%T.9b2) [symbolic]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %L.type [concrete]
// CHECK:STDOUT:   %T.0fc: %L.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.29d: type = symbolic_binding_type T, 0, %T.0fc [symbolic]
// CHECK:STDOUT:   %Interface.impl_witness.8e9: <witness> = impl_witness @T.binding.as_type.as.Interface.impl.410.%Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.410(%T.0fc) [symbolic]
// CHECK:STDOUT:   %Interface.facet.9bb: %Interface.type = facet_value %T.binding.as_type.3af, (%Interface.impl_witness.779) [symbolic]
// CHECK:STDOUT:   %Interface.facet.08d: %Interface.type = facet_value %T.binding.as_type.50b, (%Interface.impl_witness.8e2) [symbolic]
// CHECK:STDOUT:   %Interface.facet.842: %Interface.type = facet_value %T.binding.as_type.bec, (%Interface.impl_witness.84e) [symbolic]
// CHECK:STDOUT:   %Interface.facet.f9d: %Interface.type = facet_value %T.binding.as_type.29d, (%Interface.impl_witness.8e9) [symbolic]
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
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.973 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc31: %I.type = name_ref T, %T.loc27.1 [symbolic = %T.loc27.2 (constants.%T.651)]
// CHECK:STDOUT:     %T.as_type.loc31: type = facet_access_type %T.ref.loc31 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:     %.loc31: type = converted %T.ref.loc31, %T.as_type.loc31 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:     %Interface.ref.loc33: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc29: type = splice_block %I.ref.loc29 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc29: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc27.2 (constants.%T.651)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.5eb [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc42: %J.type = name_ref T, %T.loc38.1 [symbolic = %T.loc38.2 (constants.%T.612)]
// CHECK:STDOUT:     %T.as_type.loc42: type = facet_access_type %T.ref.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.50b)]
// CHECK:STDOUT:     %.loc42: type = converted %T.ref.loc42, %T.as_type.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.50b)]
// CHECK:STDOUT:     %Interface.ref.loc44: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc40: type = splice_block %J.ref.loc40 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc40: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc38.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc38.2 (constants.%T.612)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.edf [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.250 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc53: %K.type = name_ref T, %T.loc49.1 [symbolic = %T.loc49.2 (constants.%T.9b2)]
// CHECK:STDOUT:     %T.as_type.loc53: type = facet_access_type %T.ref.loc53 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.bec)]
// CHECK:STDOUT:     %.loc53: type = converted %T.ref.loc53, %T.as_type.loc53 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.bec)]
// CHECK:STDOUT:     %Interface.ref.loc55: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc51: type = splice_block %K.ref.loc51 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref.loc51: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc49.1: %K.type = symbolic_binding T, 0 [symbolic = %T.loc49.2 (constants.%T.9b2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.410 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.88f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc64: %L.type = name_ref T, %T.loc60.1 [symbolic = %T.loc60.2 (constants.%T.0fc)]
// CHECK:STDOUT:     %T.as_type.loc64: type = facet_access_type %T.ref.loc64 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.29d)]
// CHECK:STDOUT:     %.loc64: type = converted %T.ref.loc64, %T.as_type.loc64 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.29d)]
// CHECK:STDOUT:     %Interface.ref.loc66: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc62: type = splice_block %L.ref.loc62 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref.loc62: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc60.1: %L.type = symbolic_binding T, 0 [symbolic = %T.loc60.2 (constants.%T.0fc)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.973 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc75: %I.type = name_ref T, %T.loc71 [symbolic = %T.loc27.2 (constants.%T.651)]
// CHECK:STDOUT:     %T.as_type.loc75: type = facet_access_type %T.ref.loc75 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:     %.loc75: type = converted %T.ref.loc75, %T.as_type.loc75 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:     %Interface.ref.loc77: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc73: type = splice_block %I.ref.loc73 [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref.loc73: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc71: %I.type = symbolic_binding T, 0 [symbolic = %T.loc27.2 (constants.%T.651)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.5eb [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc87: %J.type = name_ref T, %T.loc83 [symbolic = %T.loc38.2 (constants.%T.612)]
// CHECK:STDOUT:     %T.as_type.loc87: type = facet_access_type %T.ref.loc87 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.50b)]
// CHECK:STDOUT:     %.loc87: type = converted %T.ref.loc87, %T.as_type.loc87 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.50b)]
// CHECK:STDOUT:     %Interface.ref.loc89: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc85: type = splice_block %J.ref.loc85 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc85: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc83: %J.type = symbolic_binding T, 0 [symbolic = %T.loc38.2 (constants.%T.612)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.edf [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.250 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc99: %K.type = name_ref T, %T.loc95 [symbolic = %T.loc49.2 (constants.%T.9b2)]
// CHECK:STDOUT:     %T.as_type.loc99: type = facet_access_type %T.ref.loc99 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.bec)]
// CHECK:STDOUT:     %.loc99: type = converted %T.ref.loc99, %T.as_type.loc99 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.bec)]
// CHECK:STDOUT:     %Interface.ref.loc101: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc97: type = splice_block %K.ref.loc97 [concrete = constants.%K.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K.ref.loc97: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc95: %K.type = symbolic_binding T, 0 [symbolic = %T.loc49.2 (constants.%T.9b2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.Interface.impl.410 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.88f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc111: %L.type = name_ref T, %T.loc107 [symbolic = %T.loc60.2 (constants.%T.0fc)]
// CHECK:STDOUT:     %T.as_type.loc111: type = facet_access_type %T.ref.loc111 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.29d)]
// CHECK:STDOUT:     %.loc111: type = converted %T.ref.loc111, %T.as_type.loc111 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.29d)]
// CHECK:STDOUT:     %Interface.ref.loc113: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %.loc109: type = splice_block %L.ref.loc109 [concrete = constants.%L.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L.ref.loc109: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc107: %L.type = symbolic_binding T, 0 [symbolic = %T.loc60.2 (constants.%T.0fc)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self.887]
// CHECK:STDOUT:   %Interface.WithSelf.decl = interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = symbolic_binding Self, 0 [symbolic = constants.%Self.44d]
// CHECK:STDOUT:   %K.WithSelf.decl = interface_with_self_decl @K [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cbc]
// CHECK:STDOUT:   %L.WithSelf.decl = interface_with_self_decl @L [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.973(%T.loc27.1: %I.type) {
// CHECK:STDOUT:   %T.loc27.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc27.2 (constants.%T.651)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3af)]
// CHECK:STDOUT:   %Interface.impl_witness.loc34.2: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.973(%T.loc27.2) [symbolic = %Interface.impl_witness.loc34.2 (constants.%Interface.impl_witness.779)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc31 as %Interface.ref.loc33 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Interface.impl.973 [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc34.1: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.973(constants.%T.651) [symbolic = %Interface.impl_witness.loc34.2 (constants.%Interface.impl_witness.779)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.5eb(%T.loc38.1: %J.type) {
// CHECK:STDOUT:   %T.loc38.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc38.2 (constants.%T.612)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc38.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.50b)]
// CHECK:STDOUT:   %Interface.impl_witness.loc45.2: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.5eb(%T.loc38.2) [symbolic = %Interface.impl_witness.loc45.2 (constants.%Interface.impl_witness.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc42 as %Interface.ref.loc44 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Interface.impl.5eb [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc45.1: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.5eb(constants.%T.612) [symbolic = %Interface.impl_witness.loc45.2 (constants.%Interface.impl_witness.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc45.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.edf(%T.loc49.1: %K.type) {
// CHECK:STDOUT:   %T.loc49.2: %K.type = symbolic_binding T, 0 [symbolic = %T.loc49.2 (constants.%T.9b2)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc49.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.bec)]
// CHECK:STDOUT:   %Interface.impl_witness.loc56.2: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.edf(%T.loc49.2) [symbolic = %Interface.impl_witness.loc56.2 (constants.%Interface.impl_witness.84e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc53 as %Interface.ref.loc55 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Interface.impl.edf [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc56.1: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.edf(constants.%T.9b2) [symbolic = %Interface.impl_witness.loc56.2 (constants.%Interface.impl_witness.84e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.Interface.impl.410(%T.loc60.1: %L.type) {
// CHECK:STDOUT:   %T.loc60.2: %L.type = symbolic_binding T, 0 [symbolic = %T.loc60.2 (constants.%T.0fc)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc60.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.29d)]
// CHECK:STDOUT:   %Interface.impl_witness.loc67.2: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.410(%T.loc60.2) [symbolic = %Interface.impl_witness.loc67.2 (constants.%Interface.impl_witness.8e9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc64 as %Interface.ref.loc66 {
// CHECK:STDOUT:     %Interface.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.Interface.impl.410 [concrete]
// CHECK:STDOUT:     %Interface.impl_witness.loc67.1: <witness> = impl_witness %Interface.impl_witness_table, @T.binding.as_type.as.Interface.impl.410(constants.%T.0fc) [symbolic = %Interface.impl_witness.loc67.2 (constants.%Interface.impl_witness.8e9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Interface.impl_witness.loc67.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Self.887) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K.WithSelf(constants.%Self.44d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L.WithSelf(constants.%Self.cbc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.973(constants.%T.651) {
// CHECK:STDOUT:   %T.loc27.2 => constants.%T.651
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3af
// CHECK:STDOUT:   %Interface.impl_witness.loc34.2 => constants.%Interface.impl_witness.779
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.5eb(constants.%T.612) {
// CHECK:STDOUT:   %T.loc38.2 => constants.%T.612
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.50b
// CHECK:STDOUT:   %Interface.impl_witness.loc45.2 => constants.%Interface.impl_witness.8e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.edf(constants.%T.9b2) {
// CHECK:STDOUT:   %T.loc49.2 => constants.%T.9b2
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.bec
// CHECK:STDOUT:   %Interface.impl_witness.loc56.2 => constants.%Interface.impl_witness.84e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.Interface.impl.410(constants.%T.0fc) {
// CHECK:STDOUT:   %T.loc60.2 => constants.%T.0fc
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.29d
// CHECK:STDOUT:   %Interface.impl_witness.loc67.2 => constants.%Interface.impl_witness.8e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.9bb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.08d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.842) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf(constants.%Interface.facet.f9d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @T.binding.as_type.as.J.impl.c4ee06.1.%J.impl_witness_table, @T.binding.as_type.as.J.impl.c4ee06.1(%T) [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %T.binding.as_type, (%J.impl_witness) [symbolic]
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
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.J.impl.c4ee06.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc15.1 [symbolic = %T.loc15.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc19: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc17: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.J.impl.c4ee06.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc27.1 [symbolic = %T.loc27.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc31: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.loc29: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc27.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.J.impl.c4ee06.1(%T.loc15.1: %I.type) {
// CHECK:STDOUT:   %T.loc15.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc15.2 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc15.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %J.impl_witness.loc22.2: <witness> = impl_witness %J.impl_witness_table, @T.binding.as_type.as.J.impl.c4ee06.1(%T.loc15.2) [symbolic = %J.impl_witness.loc22.2 (constants.%J.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc19 as %J.ref {
// CHECK:STDOUT:     %J.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.J.impl.c4ee06.1 [concrete]
// CHECK:STDOUT:     %J.impl_witness.loc22.1: <witness> = impl_witness %J.impl_witness_table, @T.binding.as_type.as.J.impl.c4ee06.1(constants.%T) [symbolic = %J.impl_witness.loc22.2 (constants.%J.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %J.impl_witness.loc22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.J.impl.c4ee06.2(%T.loc27.1: %I.type) {
// CHECK:STDOUT:   %T.loc27.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc27.2 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc27.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc31 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.J.impl.c4ee06.1(constants.%T) {
// CHECK:STDOUT:   %T.loc15.2 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %J.impl_witness.loc22.2 => constants.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.J.impl.c4ee06.2(constants.%T) {
// CHECK:STDOUT:   %T.loc27.2 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.6465e0.1: <witness> = impl_witness @C.as.I.impl.e47215.1.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.f94759.1: %I.type = facet_value %C, (%I.impl_witness.6465e0.1) [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %I.impl_witness.6465e0.2: <witness> = impl_witness @C.as.I.impl.e47215.2.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.f94759.2: %I.type = facet_value %C, (%I.impl_witness.6465e0.2) [concrete]
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl.e47215.1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.e47215.2 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc22.1: %tuple.type = tuple_literal (%C.ref.loc19, %C.ref.loc21) [concrete = constants.%tuple]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %tuple: %tuple.type = tuple_value (%C.ref.loc19, %C.ref.loc21) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc22.2: %tuple.type = converted %.loc22.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %tuple.elem0: type = tuple_access %.loc22.2, element0 [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.e47215.1: %C.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.e47215.1 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.6465e0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl.e47215.2: %tuple.elem0 as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.e47215.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.6465e0.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f94759.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f94759.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefinition_generic_regions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as.I.impl.f7b5a3.1.%I.impl_witness_table, @T.as.I.impl.f7b5a3.1(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.f7b5a3.1(%T) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.as.I.impl.A: %T.as.I.impl.A.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.f7b5a3.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.B: %T.as.I.impl.B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.f7b5a3.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.C: %T.as.I.impl.C.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.f7b5a3.2(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.D: %T.as.I.impl.D.type = struct_value () [symbolic]
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
// CHECK:STDOUT:   impl_decl @T.as.I.impl.f7b5a3.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8.2 [symbolic = %T.loc8.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc10.1: type = splice_block %.loc10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8.2: type = symbolic_binding T, 0 [symbolic = %T.loc8.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.I.impl.f7b5a3.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26.2 [symbolic = %T.loc26.1 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.loc28.1: type = splice_block %.loc28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26.2: type = symbolic_binding T, 0 [symbolic = %T.loc26.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.f7b5a3.1(%T.loc8.2: type) {
// CHECK:STDOUT:   %T.loc8.1: type = symbolic_binding T, 0 [symbolic = %T.loc8.1 (constants.%T)]
// CHECK:STDOUT:   %I.impl_witness.loc15.2: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl.f7b5a3.1(%T.loc8.1) [symbolic = %I.impl_witness.loc15.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type: type = fn_type @T.as.I.impl.A, @T.as.I.impl.f7b5a3.1(%T.loc8.1) [symbolic = %T.as.I.impl.A.type (constants.%T.as.I.impl.A.type)]
// CHECK:STDOUT:   %T.as.I.impl.A: @T.as.I.impl.f7b5a3.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = struct_value () [symbolic = %T.as.I.impl.A (constants.%T.as.I.impl.A)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.A.decl: @T.as.I.impl.f7b5a3.1.%T.as.I.impl.A.type (%T.as.I.impl.A.type) = fn_decl @T.as.I.impl.A [symbolic = @T.as.I.impl.f7b5a3.1.%T.as.I.impl.A (constants.%T.as.I.impl.A)] {} {}
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @T.as.I.impl.f7b5a3.1 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc15.1: <witness> = impl_witness %I.impl_witness_table, @T.as.I.impl.f7b5a3.1(constants.%T) [symbolic = %I.impl_witness.loc15.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .A = %T.as.I.impl.A.decl
// CHECK:STDOUT:     witness = %I.impl_witness.loc15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl.f7b5a3.2(%T.loc26.2: type) {
// CHECK:STDOUT:   %T.loc26.1: type = symbolic_binding T, 0 [symbolic = %T.loc26.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.B.type: type = fn_type @T.as.I.impl.B, @T.as.I.impl.f7b5a3.2(%T.loc26.1) [symbolic = %T.as.I.impl.B.type (constants.%T.as.I.impl.B.type)]
// CHECK:STDOUT:   %T.as.I.impl.B: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = struct_value () [symbolic = %T.as.I.impl.B (constants.%T.as.I.impl.B)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.f7b5a3.2(%T.loc26.1) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.D.type: type = fn_type @T.as.I.impl.D, @T.as.I.impl.f7b5a3.2(%T.loc26.1) [symbolic = %T.as.I.impl.D.type (constants.%T.as.I.impl.D.type)]
// CHECK:STDOUT:   %T.as.I.impl.D: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = struct_value () [symbolic = %T.as.I.impl.D (constants.%T.as.I.impl.D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %T.as.I.impl.B.decl: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.B.type (%T.as.I.impl.B.type) = fn_decl @T.as.I.impl.B [symbolic = @T.as.I.impl.f7b5a3.2.%T.as.I.impl.B (constants.%T.as.I.impl.B)] {} {}
// CHECK:STDOUT:     %T.as.I.impl.C.decl: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = fn_decl @T.as.I.impl.C [symbolic = @T.as.I.impl.f7b5a3.2.%T.as.I.impl.C (constants.%T.as.I.impl.C)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc46.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc46.2: type = converted %.loc46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc46.3: Core.Form = init_form %.loc46.2 [concrete = constants.%.262]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.as.I.impl.D.decl: @T.as.I.impl.f7b5a3.2.%T.as.I.impl.D.type (%T.as.I.impl.D.type) = fn_decl @T.as.I.impl.D [symbolic = @T.as.I.impl.f7b5a3.2.%T.as.I.impl.D (constants.%T.as.I.impl.D)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc59.2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc59.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc59.2: type = converted %.loc59.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc59.3: Core.Form = init_form %.loc59.2 [concrete = constants.%.262]
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
// CHECK:STDOUT: generic fn @T.as.I.impl.A(@T.as.I.impl.f7b5a3.1.%T.loc8.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.B(@T.as.I.impl.f7b5a3.2.%T.loc26.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.C(@T.as.I.impl.f7b5a3.2.%T.loc26.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc50.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc50.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc51: init %empty_tuple.type = converted %.loc50.1, %.loc50.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc51
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.D(@T.as.I.impl.f7b5a3.2.%T.loc26.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.as.I.impl.C.type: type = fn_type @T.as.I.impl.C, @T.as.I.impl.f7b5a3.2(%T) [symbolic = %T.as.I.impl.C.type (constants.%T.as.I.impl.C.type)]
// CHECK:STDOUT:   %T.as.I.impl.C: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = struct_value () [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:   %T.as.I.impl.C.specific_fn.loc62.2: <specific function> = specific_function %T.as.I.impl.C, @T.as.I.impl.C(%T) [symbolic = %T.as.I.impl.C.specific_fn.loc62.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc62: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = specific_constant @T.as.I.impl.f7b5a3.2.%T.as.I.impl.C.decl, @T.as.I.impl.f7b5a3.2(constants.%T) [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %C.ref: @T.as.I.impl.D.%T.as.I.impl.C.type (%T.as.I.impl.C.type) = name_ref C, %.loc62 [symbolic = %T.as.I.impl.C (constants.%T.as.I.impl.C)]
// CHECK:STDOUT:     %T.as.I.impl.C.specific_fn.loc62.1: <specific function> = specific_function %C.ref, @T.as.I.impl.C(constants.%T) [symbolic = %T.as.I.impl.C.specific_fn.loc62.2 (constants.%T.as.I.impl.C.specific_fn)]
// CHECK:STDOUT:     %T.as.I.impl.C.call: init %empty_tuple.type = call %T.as.I.impl.C.specific_fn.loc62.1()
// CHECK:STDOUT:     return %T.as.I.impl.C.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.f7b5a3.1(constants.%T) {
// CHECK:STDOUT:   %T.loc8.1 => constants.%T
// CHECK:STDOUT:   %I.impl_witness.loc15.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.A.type => constants.%T.as.I.impl.A.type
// CHECK:STDOUT:   %T.as.I.impl.A => constants.%T.as.I.impl.A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.f7b5a3.2(constants.%T) {
// CHECK:STDOUT:   %T.loc26.1 => constants.%T
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

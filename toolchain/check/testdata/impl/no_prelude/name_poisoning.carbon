// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/name_poisoning.carbon

// --- using_poisoned_name_in_impl.carbon

library "[[@TEST_NAME]]";

interface I {};

namespace N;
// Use `package.I` and poison `N.I`.
fn N.F1(x:! I);

class N.C {
  extend impl as I {
  }
}

// --- fail_impl_function_poisoned.carbon

library "[[@TEST_NAME]]";

interface I {
  fn A(x: Self);
  fn B();
}

class B {
  impl as I {
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+3]]:13: error: name `B` used before it was declared [NameUseBeforeDecl]
    // CHECK:STDERR:     fn A(x: B);
    // CHECK:STDERR:             ^
    fn A(x: B);
    // TODO: Avoid ImplMissingFunction for functions that were declared after they were poisoned.
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE+11]]:8: note: declared here [NameUseBeforeDeclNote]
    // CHECK:STDERR:     fn B();
    // CHECK:STDERR:        ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE-10]]:3: error: missing implementation of B in impl of interface I [ImplMissingFunction]
    // CHECK:STDERR:   impl as I {
    // CHECK:STDERR:   ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_function_poisoned.carbon:[[@LINE-17]]:3: note: associated function B declared here [AssociatedFunctionHere]
    // CHECK:STDERR:   fn B();
    // CHECK:STDERR:   ^~~~~~~
    // CHECK:STDERR:
    fn B();
  }
}

// CHECK:STDOUT: --- using_poisoned_name_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %x: %I.type = bind_symbolic_name x, 0 [symbolic]
// CHECK:STDOUT:   %x.patt: %I.type = symbolic_binding_pattern x, 0 [symbolic]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %x.patt.loc8_9.1: %I.type = symbolic_binding_pattern x, 0 [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:     %x.param_patt: %I.type = value_param_pattern %x.patt.loc8_9.1, runtime_param<none> [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %I.type = value_param runtime_param<none>
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %x.loc8_9.1: %I.type = bind_symbolic_name x, 0, %x.param [symbolic = %x.loc8_9.2 (constants.%x)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   extend @impl.%I.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F1(%x.loc8_9.1: %I.type) {
// CHECK:STDOUT:   %x.loc8_9.2: %I.type = bind_symbolic_name x, 0 [symbolic = %x.loc8_9.2 (constants.%x)]
// CHECK:STDOUT:   %x.patt.loc8_9.2: %I.type = symbolic_binding_pattern x, 0 [symbolic = %x.patt.loc8_9.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: %I.type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1(constants.%x) {
// CHECK:STDOUT:   %x.loc8_9.2 => constants.%x
// CHECK:STDOUT:   %x.patt.loc8_9.2 => constants.%x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_function_poisoned.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %A.type.edf: type = fn_type @A.1 [concrete]
// CHECK:STDOUT:   %A.ab0: %A.type.edf = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%A.decl [concrete]
// CHECK:STDOUT:   %B.type.785: type = fn_type @B.1 [concrete]
// CHECK:STDOUT:   %B.6d4: %B.type.785 = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%B.decl [concrete]
// CHECK:STDOUT:   %B.fa3: type = class_type @B.3 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%A.decl, <error>) [concrete]
// CHECK:STDOUT:   %A.type.9b9: type = fn_type @A.2 [concrete]
// CHECK:STDOUT:   %A.71f: %A.type.9b9 = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.52d: type = fn_type @B.2 [concrete]
// CHECK:STDOUT:   %B.30a: %B.type.52d = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %B.fa3, %impl_witness [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B.3 [concrete = constants.%B.fa3] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.decl: %A.type.edf = fn_decl @A.1 [concrete = constants.%A.ab0] {
// CHECK:STDOUT:     %x.patt: @A.1.%Self.as_type.loc5_11.1 (%Self.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @A.1.%Self.as_type.loc5_11.1 (%Self.as_type) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @A.1.%Self.as_type.loc5_11.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_11.1: type = splice_block %.loc5_11.2 [symbolic = %Self.as_type.loc5_11.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_11.2: type = converted %Self.ref, %Self.as_type.loc5_11.2 [symbolic = %Self.as_type.loc5_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @A.1.%Self.as_type.loc5_11.1 (%Self.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %A.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %B.decl: %B.type.785 = fn_decl @B.1 [concrete = constants.%B.6d4] {} {}
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %B.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = %assoc0
// CHECK:STDOUT:   .B = %assoc1
// CHECK:STDOUT:   witness = (%A.decl, %B.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %A.decl: %A.type.9b9 = fn_decl @A.2 [concrete = constants.%A.71f] {
// CHECK:STDOUT:     %x.patt: %B.fa3 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %B.fa3 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %B.fa3 = value_param runtime_param0
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.fa3]
// CHECK:STDOUT:     %x: %B.fa3 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type.52d = fn_decl @B.2 [concrete = constants.%B.30a] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT:   witness = @B.3.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.3 {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.fa3 [concrete = constants.%B.fa3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%A.decl, <error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.fa3
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @A.1.%Self.as_type.loc5_11.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2(%x.param_patt: %B.fa3);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_11.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_11.1 => constants.%B.fa3
// CHECK:STDOUT: }
// CHECK:STDOUT:

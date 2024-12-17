// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon

// --- fail_derived_to_base.carbon

fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";

base class B {}

class A(N:! i32) {
  extend base: B;

  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+12]]:14: error: cannot implicitly convert from `i32` to `Core.Int(N)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+9]]:14: note: type `i32` does not implement interface `Core.ImplicitAs(Core.Int(N))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:              ^
  // CHECK:STDERR: /usr/local/google/home/joshl/.cache/bazel/_bazel_joshl/6f495ed483b744d19e67a46396fa6c5a/execroot/_main/bazel-out/k8-dbg/bin/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/core/prelude/types/int.carbon:21:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE-16]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~
  var n: Int(N);
}

fn F(a: A(0)*) {
  let b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type.1: type = fn_type @Int.1 [template]
// CHECK:STDOUT:   %Int.1: %Int.type.1 = struct_value () [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int.2: type = class_type @Int.2, @Int.2(%N.1) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int.2, @Int.2(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%N.2) [symbolic]
// CHECK:STDOUT:   %A.elem.1: type = unbound_element_type %A.1, %B [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.2)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %ImplicitAs.type.4 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.3: <witness> = require_complete_type %Int.2 [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.2(%N.1) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.4), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N.2, %Convert.4 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.3(%N.1) [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.7: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %Int.2 = call %Convert.specific_fn.2(%int_0) [symbolic]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%int.convert_checked) [symbolic]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %A.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %B [template]
// CHECK:STDOUT:   %A.elem.2: type = unbound_element_type %A.2, %B [symbolic]
// CHECK:STDOUT:   %Int.3: type = class_type @Int.2, @Int.2(%int.convert_checked) [symbolic]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.3, @impl.2(%int.convert_checked) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.convert_checked, %Convert.13 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.3(%int.convert_checked) [symbolic]
// CHECK:STDOUT:   %require_complete.6: <witness> = require_complete_type %Int.3 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.3)> [symbolic]
// CHECK:STDOUT:   %require_complete.7: <witness> = require_complete_type %ImplicitAs.type.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type.1 = fn_decl @Int.1 [template = constants.%Int.1] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int_literal.make_type, %.loc2_27.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int.2, @Int.2(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %ptr.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.11 = impl_witness_access constants.%impl_witness.7, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N.1) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:     %int.convert_checked: init %Int.2 = call %Convert.specific_fn(%int_0) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:     %.loc24_12.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:     %.loc24_12.2: %i32 = converted %int_0, %.loc24_12.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%int.convert_checked) [symbolic = constants.%A.2]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %A.2 [symbolic = constants.%ptr.2]
// CHECK:STDOUT:     %a.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %ptr.2 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.2) [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type @A.%A (%A.1), %B [symbolic = %A.elem (constants.%A.elem.1)]
// CHECK:STDOUT:   %N.loc21: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc21 (constants.%N.1)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.2(%N.loc21) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @A.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %Convert.bound.loc21_14.2: <bound method> = bound_method %N.loc6_9.2, %Convert [symbolic = %Convert.bound.loc21_14.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_14.2: <specific function> = specific_function %Convert.bound.loc21_14.2, @Convert.3(%N.loc21) [symbolic = %Convert.specific_fn.loc21_14.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %Int: type = class_type @Int.2, @Int.2(%N.loc21) [symbolic = %Int (constants.%Int.2)]
// CHECK:STDOUT:   %require_complete.loc21_14.1: <witness> = require_complete_type @A.%Int (%Int.2) [symbolic = %require_complete.loc21_14.1 (constants.%require_complete.3)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Int)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %require_complete.loc21_14.2: <witness> = require_complete_type @A.%ImplicitAs.type (%ImplicitAs.type.4) [symbolic = %require_complete.loc21_14.2 (constants.%require_complete.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem (%A.elem.1) = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:     %Int.ref: %Int.type.1 = name_ref Int, file.%Int.decl [template = constants.%Int.1]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc6_9.1 [symbolic = %N.loc6_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:     %Convert.bound.loc21_14.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound.loc21_14.2 (constants.%Convert.bound.1)]
// CHECK:STDOUT:     %Convert.specific_fn.loc21_14.1: <specific function> = specific_function %Convert.bound.loc21_14.1, @Convert.3(constants.%N.1) [symbolic = %Convert.specific_fn.loc21_14.2 (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:     %.loc21_14.1: @A.%Int (%Int.2) = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.loc21_14.1(<error>) [template = <error>]
// CHECK:STDOUT:     %.loc21_14.2: Core.IntLiteral = value_of_initializer %int.convert_checked [template = <error>]
// CHECK:STDOUT:     %.loc21_14.3: Core.IntLiteral = converted %N.ref, %.loc21_14.2 [template = <error>]
// CHECK:STDOUT:     %int.make_type_signed: init type = call %Int.ref(%.loc21_14.3) [template = <error>]
// CHECK:STDOUT:     %.loc21_15.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:     %.loc21_15.2: type = converted %int.make_type_signed, %.loc21_15.1 [template = <error>]
// CHECK:STDOUT:     %.loc21_8: <error> = field_decl n, element1 [template]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.1
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .n = %.loc21_8
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.1(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %ptr.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %ptr.loc25: type = ptr_type %B [template = constants.%ptr.3]
// CHECK:STDOUT:   %a.ref: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc25_16.1: ref %A.2 = deref %a.ref
// CHECK:STDOUT:   %.loc25_16.2: ref %B = class_element_access %.loc25_16.1, element0
// CHECK:STDOUT:   %addr: %ptr.3 = addr_of %.loc25_16.2
// CHECK:STDOUT:   %.loc25_16.3: %ptr.3 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.3 = bind_name b, %.loc25_16.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%N.loc6_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int.convert_checked) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int.convert_checked
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int.convert_checked
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2
// CHECK:STDOUT:   %N.loc21 => constants.%int.convert_checked
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.13
// CHECK:STDOUT:   %Convert => constants.%Convert.13
// CHECK:STDOUT:   %Convert.bound.loc21_14.2 => constants.%Convert.bound.3
// CHECK:STDOUT:   %Convert.specific_fn.loc21_14.2 => constants.%Convert.specific_fn.3
// CHECK:STDOUT:   %Int => constants.%Int.3
// CHECK:STDOUT:   %require_complete.loc21_14.1 => constants.%require_complete.6
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %require_complete.loc21_14.2 => constants.%require_complete.7
// CHECK:STDOUT: }
// CHECK:STDOUT:

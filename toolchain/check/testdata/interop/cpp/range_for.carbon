// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:35: error: cannot implicitly convert non-type value of type `Self.(TODO: element 0 in incomplete Self as CppRangeForIterate).(CppUnsafeDeref.Result)` into type implementing `OptionalStorage` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:   fn Next(iterator: Iterator*) -> Optional(ElementType);
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:35: note: type `Self.(TODO: element 0 in incomplete Self as CppRangeForIterate).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(OptionalStorage)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   fn Next(iterator: Iterator*) -> Optional(ElementType);
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:26:16: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Optional(T:! OptionalStorage) {
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:47:24: error: cannot implicitly convert non-type value of type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` into type implementing `Copy & Destroy` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:   where .ElementType = R.ElementType and .CursorType = (R.Iterator, R.Sentinel)
// CHECK:STDERR:                        ^~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:47:24: note: type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(Copy & Destroy)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   where .ElementType = R.ElementType and .CursorType = (R.Iterator, R.Sentinel)
// CHECK:STDERR:                        ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:53:68: error: cannot implicitly convert non-type value of type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` into type implementing `OptionalStorage` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:   fn Next[unused self: Self](cursor: (R.Iterator, R.Sentinel)*) -> Optional(R.ElementType) {
// CHECK:STDERR:                                                                    ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:53:68: note: type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(OptionalStorage)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   fn Next[unused self: Self](cursor: (R.Iterator, R.Sentinel)*) -> Optional(R.ElementType) {
// CHECK:STDERR:                                                                    ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:26:16: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Optional(T:! OptionalStorage) {
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:55:28: error: cannot implicitly convert non-type value of type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` into type implementing `OptionalStorage` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some((cursor->0).(CppUnsafeDeref.Op)());
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:55:28: note: type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(OptionalStorage)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some((cursor->0).(CppUnsafeDeref.Op)());
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:26:16: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Optional(T:! OptionalStorage) {
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:55:7: error: `returned var` declaration in function with non-initializing return form [ReturnedVarNotInit]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some((cursor->0).(CppUnsafeDeref.Op)());
// CHECK:STDERR:       ^~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon: note: return form declared here [ReturnFormHereNote]
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:55:54: error: cannot implicitly convert non-type value of type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` into type implementing `OptionalStorage` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some((cursor->0).(CppUnsafeDeref.Op)());
// CHECK:STDERR:                                                      ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:55:54: note: type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(OptionalStorage)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some((cursor->0).(CppUnsafeDeref.Op)());
// CHECK:STDERR:                                                      ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:26:16: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Optional(T:! OptionalStorage) {
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:59:14: error: cannot implicitly convert non-type value of type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` into type implementing `OptionalStorage` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR:       return Optional(R.ElementType).None();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:59:14: note: type `R.(CppRangeForIterate.Iterator).(CppUnsafeDeref.Result)` does not implement interface `ImplicitAs(OptionalStorage)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:       return Optional(R.ElementType).None();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:7:1: in import [InImport]
// CHECK:STDERR: {{.*}}/prelude/types/optional.carbon:26:16: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class Optional(T:! OptionalStorage) {
// CHECK:STDERR:                ^
// CHECK:STDERR:

// --- begin_end_return_same_class_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
template<class T>
class Iterable {
  struct Iterator {
    auto operator*() const -> T&;
    auto operator++() -> Iterator&;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  auto begin() -> Iterator;
  auto end() -> Iterator;
};
''';

fn Sum(v: Cpp.Iterable(i32)) -> i32 {
  returned var result: i32 = 0;

  //@dump-sem-ir-begin
  // CHECK:STDERR: begin_end_return_same_class_type.carbon:[[@LINE+8]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.Iterable` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (var i: i32 in v) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: begin_end_return_same_class_type.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.Iterable` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (var i: i32 in v) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (var i: i32 in v) {
    result += i;
  }
  //@dump-sem-ir-end

  return var;
}

// CHECK:STDOUT: --- begin_end_return_same_class_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Iterable: type = class_type @Iterable [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.fc6021.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.299: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.fc6021.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.214: %facet_type.299 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.binding.as_type.as.ImplicitAs.impl.Convert.type.35d: type = fn_type @From.binding.as_type.as.ImplicitAs.impl.Convert, @From.binding.as_type.as.ImplicitAs.impl(%To, %From.214) [symbolic]
// CHECK:STDOUT:   %From.binding.as_type.as.ImplicitAs.impl.Convert.a11: %From.binding.as_type.as.ImplicitAs.impl.Convert.type.35d = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.794: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.39a54f.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.fc6021.1)> [symbolic]
// CHECK:STDOUT:   %U.354: %ImplicitAs.type.39a54f.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.b24e94.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.28caa4.1: %Int.as.AddAssignWith.impl.Op.type.b24e94.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.b24e94.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.354) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.28caa4.2: %Int.as.AddAssignWith.impl.Op.type.b24e94.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.fc9: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.a1b: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.5e9: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.86d: %Int.as.AnyInt.impl.AsInt.type.5e9 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.c17: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.fc9 = facet_value %i32, (%custom_witness.a1b, %AnyInt.impl_witness.c17) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e18: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.904, @From.binding.as_type.as.ImplicitAs.impl(%int_32, %facet_value) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3c3: %ImplicitAs.type.e8c = facet_value %i32, (%ImplicitAs.impl_witness.e18) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.eb7: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.3c3) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.00ea6e.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.3c3) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.b1120a.1: %Int.as.AddAssignWith.impl.Op.type.00ea6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.00ea6e.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.3c3) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.b1120a.2: %Int.as.AddAssignWith.impl.Op.type.00ea6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.794 = facet_value %i32, (%AddAssignWith.impl_witness.eb7) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.b88: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.51b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.b88, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3d93b4.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.b1120a.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.3c3) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3d93b4.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.b1120a.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.3c3) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.8ae8: @From.binding.as_type.as.ImplicitAs.impl.%From.binding.as_type.as.ImplicitAs.impl.Convert.type (%From.binding.as_type.as.ImplicitAs.impl.Convert.type.35d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.binding.as_type.as.ImplicitAs.impl.%From.binding.as_type.as.ImplicitAs.impl.Convert (constants.%From.binding.as_type.as.ImplicitAs.impl.Convert.a11)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.904 = impl_witness_table (%Core.import_ref.8ae8), @From.binding.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.52d: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.b24e94.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.28caa4.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.52d), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d21: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.b24e94.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.28caa4.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.928: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.5e9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.86d)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.928), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sum(%v.param: %Iterable) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.7ce = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: %Iterable = name_ref v, %v
// CHECK:STDOUT:   %var: ref <error> = var invalid [concrete = <error>]
// CHECK:STDOUT:   assign %var, <error>
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: <error> = addr_of %var [concrete = <error>]
// CHECK:STDOUT:   %HasValue.ref: <error> = name_ref HasValue, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %i.var: ref %i32 = var %i.var_patt
// CHECK:STDOUT:   %Get.ref: <error> = name_ref Get, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %i32.loc46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i: ref %i32 = ref_binding i, %i.var
// CHECK:STDOUT:   %result.ref: ref %i32 = name_ref result, %result
// CHECK:STDOUT:   %i.ref: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc47: %.51b = impl_witness_access constants.%AddAssignWith.impl_witness.eb7, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.b1120a.2]
// CHECK:STDOUT:   %bound_method.loc47_12.1: <bound method> = bound_method %result.ref, %impl.elem0.loc47
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.3c3) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3d93b4.1]
// CHECK:STDOUT:   %bound_method.loc47_12.2: <bound method> = bound_method %result.ref, %specific_fn.loc47
// CHECK:STDOUT:   %.loc47_15: %i32 = acquire_value %i.ref
// CHECK:STDOUT:   %.loc47_12: %Int.as.AddAssignWith.impl.Op.type.00ea6e.1 = specific_constant imports.%Core.Op.d21, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.3c3) [concrete = constants.%Int.as.AddAssignWith.impl.Op.b1120a.1]
// CHECK:STDOUT:   %Op.ref: %Int.as.AddAssignWith.impl.Op.type.00ea6e.1 = name_ref Op, %.loc47_12 [concrete = constants.%Int.as.AddAssignWith.impl.Op.b1120a.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %result.ref, %Op.ref
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.3c3) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3d93b4.2]
// CHECK:STDOUT:   %bound_method.loc47_12.3: <bound method> = bound_method %result.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc47_12.3(%result.ref, %.loc47_15)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc46: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc46: init %empty_tuple.type = call %Destroy.Op.bound.loc46(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:

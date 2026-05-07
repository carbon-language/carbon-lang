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

// --- begin_end_return_same_class_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class Iter {
public:
  struct Iterator {
    auto operator*() const -> int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
  auto begin() const -> Iterator;
  auto end() const -> Iterator;
};
''';

fn CppRangeForIterate[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  // CHECK:STDERR: begin_end_return_same_class_type.carbon:[[@LINE+8]]:3: error: cannot access member of interface `Core.Destroy` in type `R.(Core.CppRangeForIterate.Sentinel)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   r.End();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: begin_end_return_same_class_type.carbon:[[@LINE-5]]:3: error: cannot access member of interface `Core.Destroy` in type `R.(Core.CppRangeForIterate.Iterator)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   r.Begin();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  r.End();
  //@dump-sem-ir-end
}

fn Test(v: Cpp.Iter) {
  CppRangeForIterate(v);
}

// CHECK:STDOUT: --- begin_end_return_same_class_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type.093: type = facet_type <@CppRangeForIterate.1> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type.093 = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type.093 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate.1 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate.1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate.1 where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate.1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type.093 = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Iter: type = class_type @Iter [concrete]
// CHECK:STDOUT:   %pattern_type.221: type = pattern_type %Iter [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Iter.Begin.type: type = fn_type @Iter.Begin [concrete]
// CHECK:STDOUT:   %Iter.Begin: %Iter.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %Iter.End.type: type = fn_type @Iter.End [concrete]
// CHECK:STDOUT:   %Iter.End: %Iter.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d7a: <witness> = custom_witness (%Iterator, %Iterator, %Iter.Begin, %Iter.End), @CppRangeForIterate.1 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.bf3: %CppRangeForIterate.type.093 = facet_value %Iter, (%custom_witness.d7a) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.164: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %Iter, (%custom_witness.d7a, %custom_witness.164) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.ac4: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.bf3) [concrete]
// CHECK:STDOUT:   %.026: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.ac4, %CppRangeForIterate.facet.bf3 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2a8: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.bf3) [concrete]
// CHECK:STDOUT:   %.596: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2a8, %CppRangeForIterate.facet.bf3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CppRangeForIterate.loc18(%R.loc18_24.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc18_24.1, @CppRangeForIterate.1 [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type.093 = facet_value %R.as_type.loc18_123.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc20_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc20_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc20_4.2: @CppRangeForIterate.loc18.%.loc20_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc20_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_4.2: <specific function> = specific_impl_function %impl.elem2.loc20_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc20_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc29_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc29_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc29_4.2: @CppRangeForIterate.loc18.%.loc29_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc29_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc29_4.2: <specific function> = specific_impl_function %impl.elem3.loc29_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc29_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @CppRangeForIterate.loc18.%R.as_type.loc18_123.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc20: @CppRangeForIterate.loc18.%R.as_type.loc18_123.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc20_4.1: @CppRangeForIterate.loc18.%.loc20_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc20_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc20_4: <bound method> = bound_method %r.ref.loc20, %impl.elem2.loc20_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_4.1: <specific function> = specific_impl_function %impl.elem2.loc20_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc20_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc20_11: <bound method> = bound_method %r.ref.loc20, %specific_impl_fn.loc20_4.1
// CHECK:STDOUT:     %.loc20_11.1: ref @CppRangeForIterate.loc18.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @CppRangeForIterate.loc18.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc20_11.1 = call %bound_method.loc20_11(%r.ref.loc20)
// CHECK:STDOUT:     %.loc20_11.2: ref @CppRangeForIterate.loc18.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc20_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc29: @CppRangeForIterate.loc18.%R.as_type.loc18_123.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc29_4.1: @CppRangeForIterate.loc18.%.loc29_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc29_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc29_4: <bound method> = bound_method %r.ref.loc29, %impl.elem3.loc29_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc29_4.1: <specific function> = specific_impl_function %impl.elem3.loc29_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc29_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc29_9: <bound method> = bound_method %r.ref.loc29, %specific_impl_fn.loc29_4.1
// CHECK:STDOUT:     %.loc29_9.1: ref @CppRangeForIterate.loc18.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @CppRangeForIterate.loc18.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc29_9.1 = call %bound_method.loc29_9(%r.ref.loc29)
// CHECK:STDOUT:     %.loc29_9.2: ref @CppRangeForIterate.loc18.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc29_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CppRangeForIterate.loc18(constants.%R) {
// CHECK:STDOUT:   %R.loc18_24.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc18_123.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CppRangeForIterate.loc18(constants.%facet_value) {
// CHECK:STDOUT:   %R.loc18_24.1 => constants.%facet_value
// CHECK:STDOUT:   %R.as_type.loc18_123.1 => constants.%Iter
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.221
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.d7a
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.bf3
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.ac4
// CHECK:STDOUT:   %.loc20_4 => constants.%.026
// CHECK:STDOUT:   %impl.elem2.loc20_4.2 => constants.%Iter.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc20_4.2 => constants.%Iter.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.2a8
// CHECK:STDOUT:   %.loc29_4 => constants.%.596
// CHECK:STDOUT:   %impl.elem3.loc29_4.2 => constants.%Iter.End
// CHECK:STDOUT:   %specific_impl_fn.loc29_4.2 => constants.%Iter.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/non_trivial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/non_trivial.carbon

// --- core.carbon

package Core library "core";

export import library "prelude/parts/array";

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a vaule's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// --- cpp_types.h

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

// --- todo_fail_cpp_class_types.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestNonTrivialDestructor(x: Cpp.NonTrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

inline Cpp '''c++
template <typename T>
struct ParameterizedDataMember {
  T t;
};
''';

fn TestHolderOfNonTrivialDestructor(x: Cpp.ParameterizedDataMember(Cpp.NonTrivialDestructor))  {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

class HasNonTrivialDestroy {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn TestHolderOfCarbonNonTrivial(x: Cpp.ParameterizedDataMember(HasNonTrivialDestroy)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_carbon_class_types.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class HasNonTrivialDestructorMember {
  var n: Cpp.NonTrivialDestructor;
}

fn TestNonTrivialDestructorMember(x: HasNonTrivialDestructorMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

class DerivedFromNonTrivialDestructor {
  extend base: Cpp.NonTrivialDestructor;
}

fn TestDerivedFromNonTrivialDestructor(x: DerivedFromNonTrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

class GenericField(T: Core.Destroy) {
  var t: T;
}

fn TestGenericField(x: GenericField(Cpp.NonTrivialDestructor))  {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- bidirectional_interop.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class HasNonTrivialDestructorMember {
  var n: Cpp.NonTrivialDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsNonTrivialDestructor {
  Carbon::HasNonTrivialDestructorMember c;
};
''';

fn TestBidirectionalInterop(x: Cpp.CppHoldsCarbonHoldsNonTrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_class_with_virtual_function.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
struct VirtualFunction {
  virtual void F();
};
''';

fn TestVirtualFunction(x: Cpp.VirtualFunction) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
}

// --- todo_fail_class_with_virtual_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
struct VirtualDestructor {
  virtual ~VirtualDestructor();
};
''';

fn TestVirtualDestructor(x: Cpp.VirtualDestructor)  {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
}

// CHECK:STDOUT: --- todo_fail_cpp_class_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor.type: type = fn_type @NonTrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor: %NonTrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1878.1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1878.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9f1: %Destroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1878.1, %DestroyImpl.impl_witness.579) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.f04: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.9f1) [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.f1c: type = class_type @ParameterizedDataMember.1 [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.type.634098.1: type = fn_type @ParameterizedDataMember.cpp_destructor.1 [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.5dc792.1: %ParameterizedDataMember.cpp_destructor.type.634098.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1878.2: <witness> = custom_witness (%ParameterizedDataMember.cpp_destructor.5dc792.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9ae: %SubobjectDestroy.type = facet_value %ParameterizedDataMember.f1c, (%custom_witness.fe1878.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a74: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9ae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.165: %Destroy.type = facet_value %ParameterizedDataMember.f1c, (%custom_witness.fe1878.2, %DestroyImpl.impl_witness.a74) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.373: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.165) [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.b54: type = class_type @ParameterizedDataMember.2 [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.type.634098.2: type = fn_type @ParameterizedDataMember.cpp_destructor.2 [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.5dc792.2: %ParameterizedDataMember.cpp_destructor.type.634098.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1878.3: <witness> = custom_witness (%ParameterizedDataMember.cpp_destructor.5dc792.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ba5: %SubobjectDestroy.type = facet_value %ParameterizedDataMember.b54, (%custom_witness.fe1878.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e68: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ba5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.df1: %Destroy.type = facet_value %ParameterizedDataMember.b54, (%custom_witness.fe1878.3, %DestroyImpl.impl_witness.e68) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.328: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.df1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialDestructor(%x.param: %NonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1878.1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1878.1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.9f1) [concrete = constants.%CheckImplsDestroy.specific_fn.f04]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHolderOfNonTrivialDestructor(%x.param: %ParameterizedDataMember.f1c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ParameterizedDataMember.f1c = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc25_22.1: %Destroy.type = facet_value constants.%ParameterizedDataMember.f1c, (constants.%custom_witness.fe1878.2, constants.%DestroyImpl.impl_witness.a74) [concrete = constants.%Destroy.facet.165]
// CHECK:STDOUT:   %.loc25_22.1: %Destroy.type = converted constants.%ParameterizedDataMember.f1c, %Destroy.facet.loc25_22.1 [concrete = constants.%Destroy.facet.165]
// CHECK:STDOUT:   %Destroy.facet.loc25_22.2: %Destroy.type = facet_value constants.%ParameterizedDataMember.f1c, (constants.%custom_witness.fe1878.2, constants.%DestroyImpl.impl_witness.a74) [concrete = constants.%Destroy.facet.165]
// CHECK:STDOUT:   %.loc25_22.2: %Destroy.type = converted constants.%ParameterizedDataMember.f1c, %Destroy.facet.loc25_22.2 [concrete = constants.%Destroy.facet.165]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.165) [concrete = constants.%CheckImplsDestroy.specific_fn.373]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHolderOfCarbonNonTrivial(%x.param: %ParameterizedDataMember.b54) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ParameterizedDataMember.b54 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc40_22.1: %Destroy.type = facet_value constants.%ParameterizedDataMember.b54, (constants.%custom_witness.fe1878.3, constants.%DestroyImpl.impl_witness.e68) [concrete = constants.%Destroy.facet.df1]
// CHECK:STDOUT:   %.loc40_22.1: %Destroy.type = converted constants.%ParameterizedDataMember.b54, %Destroy.facet.loc40_22.1 [concrete = constants.%Destroy.facet.df1]
// CHECK:STDOUT:   %Destroy.facet.loc40_22.2: %Destroy.type = facet_value constants.%ParameterizedDataMember.b54, (constants.%custom_witness.fe1878.3, constants.%DestroyImpl.impl_witness.e68) [concrete = constants.%Destroy.facet.df1]
// CHECK:STDOUT:   %.loc40_22.2: %Destroy.type = converted constants.%ParameterizedDataMember.b54, %Destroy.facet.loc40_22.2 [concrete = constants.%Destroy.facet.df1]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.df1) [concrete = constants.%CheckImplsDestroy.specific_fn.328]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_carbon_class_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestructorMember: type = class_type @HasNonTrivialDestructorMember [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasNonTrivialDestructorMember.elem: type = unbound_element_type %HasNonTrivialDestructorMember, %NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.n.a8b: type = struct_type {.n: %NonTrivialDestructor} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor.type: type = fn_type @NonTrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor: %NonTrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9f1: %Destroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.579) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b99: %SubobjectDestroy.type = facet_value %HasNonTrivialDestructorMember, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.949: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b99) [concrete]
// CHECK:STDOUT:   %Destroy.facet.245: %Destroy.type = facet_value %HasNonTrivialDestructorMember, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.949) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.61d: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.245) [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialDestructor: type = class_type @DerivedFromNonTrivialDestructor [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialDestructor.elem: type = unbound_element_type %DerivedFromNonTrivialDestructor, %NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.e72: type = struct_type {.base: %NonTrivialDestructor} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.66d: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialDestructor, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7ab: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.66d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a7d: %Destroy.type = facet_value %DerivedFromNonTrivialDestructor, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.7ab) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.a36: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.a7d) [concrete]
// CHECK:STDOUT:   %GenericField.829: type = class_type @GenericField, @GenericField(%Destroy.facet.9f1) [concrete]
// CHECK:STDOUT:   %GenericField.elem.367: type = unbound_element_type %GenericField.829, %NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.t.7ae: type = struct_type {.t: %NonTrivialDestructor} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc39_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e81: %SubobjectDestroy.type = facet_value %GenericField.829, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.274: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e81) [concrete]
// CHECK:STDOUT:   %Destroy.facet.471: %Destroy.type = facet_value %GenericField.829, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.274) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.e95: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.471) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialDestructorMember(%x.param: %HasNonTrivialDestructorMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %HasNonTrivialDestructorMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%HasNonTrivialDestructorMember, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.949) [concrete = constants.%Destroy.facet.245]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%HasNonTrivialDestructorMember, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.245]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%HasNonTrivialDestructorMember, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.949) [concrete = constants.%Destroy.facet.245]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%HasNonTrivialDestructorMember, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.245]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.245) [concrete = constants.%CheckImplsDestroy.specific_fn.61d]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %struct_type.n.a8b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %HasNonTrivialDestructorMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %HasNonTrivialDestructorMember.elem = name_ref n, @HasNonTrivialDestructorMember.%field_decl [concrete = @HasNonTrivialDestructorMember.%field_decl]
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromNonTrivialDestructor(%x.param: %DerivedFromNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromNonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialDestructor, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %.loc26_22.1: %Destroy.type = converted constants.%DerivedFromNonTrivialDestructor, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialDestructor, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%DerivedFromNonTrivialDestructor, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.a7d) [concrete = constants.%CheckImplsDestroy.specific_fn.a36]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.1(%self.param: ref %struct_type.base.e72) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_22.1: ref %NonTrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc26_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_22.2(%self.param: ref %DerivedFromNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialDestructor.elem = name_ref base, @DerivedFromNonTrivialDestructor.%.loc21 [concrete = @DerivedFromNonTrivialDestructor.%.loc21]
// CHECK:STDOUT:   %.loc26_22.1: ref %NonTrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc26_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc26_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc26_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc26_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc26_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc26_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc26_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericField(%x.param: %GenericField.829) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %GenericField.829 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc39_22.1: %Destroy.type = facet_value constants.%GenericField.829, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.274) [concrete = constants.%Destroy.facet.471]
// CHECK:STDOUT:   %.loc39_22.1: %Destroy.type = converted constants.%GenericField.829, %Destroy.facet.loc39_22.1 [concrete = constants.%Destroy.facet.471]
// CHECK:STDOUT:   %Destroy.facet.loc39_22.2: %Destroy.type = facet_value constants.%GenericField.829, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.274) [concrete = constants.%Destroy.facet.471]
// CHECK:STDOUT:   %.loc39_22.2: %Destroy.type = converted constants.%GenericField.829, %Destroy.facet.loc39_22.2 [concrete = constants.%Destroy.facet.471]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.471) [concrete = constants.%CheckImplsDestroy.specific_fn.e95]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc39_22.1(%self.param: ref %struct_type.t.7ae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc39_22.1: ref %NonTrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc39_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc39_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc39_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc39_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc39_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc39_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc39_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc39_22.2(%self.param: ref %GenericField.829) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %GenericField.elem.367 = name_ref t, @GenericField.%field_decl [concrete = @GenericField.%field_decl]
// CHECK:STDOUT:   %.loc39_22.1: ref %NonTrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc39_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc39_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc39_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc39_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc39_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc39_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc39_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bidirectional_interop.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %CppHoldsCarbonHoldsNonTrivialDestructor: type = class_type @CppHoldsCarbonHoldsNonTrivialDestructor [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %CppHoldsCarbonHoldsNonTrivialDestructor.cpp_destructor.type: type = fn_type @CppHoldsCarbonHoldsNonTrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %CppHoldsCarbonHoldsNonTrivialDestructor.cpp_destructor: %CppHoldsCarbonHoldsNonTrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1878.2: <witness> = custom_witness (%CppHoldsCarbonHoldsNonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7c4: %SubobjectDestroy.type = facet_value %CppHoldsCarbonHoldsNonTrivialDestructor, (%custom_witness.fe1878.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9d9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7c4) [concrete]
// CHECK:STDOUT:   %Destroy.facet.dd9: %Destroy.type = facet_value %CppHoldsCarbonHoldsNonTrivialDestructor, (%DestroyImpl.impl_witness.9d9, %custom_witness.fe1878.2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.dd9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBidirectionalInterop(%x.param: %CppHoldsCarbonHoldsNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %CppHoldsCarbonHoldsNonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc19_22.1: %Destroy.type = facet_value constants.%CppHoldsCarbonHoldsNonTrivialDestructor, (constants.%DestroyImpl.impl_witness.9d9, constants.%custom_witness.fe1878.2) [concrete = constants.%Destroy.facet.dd9]
// CHECK:STDOUT:   %.loc19_22.1: %Destroy.type = converted constants.%CppHoldsCarbonHoldsNonTrivialDestructor, %Destroy.facet.loc19_22.1 [concrete = constants.%Destroy.facet.dd9]
// CHECK:STDOUT:   %Destroy.facet.loc19_22.2: %Destroy.type = facet_value constants.%CppHoldsCarbonHoldsNonTrivialDestructor, (constants.%DestroyImpl.impl_witness.9d9, constants.%custom_witness.fe1878.2) [concrete = constants.%Destroy.facet.dd9]
// CHECK:STDOUT:   %.loc19_22.2: %Destroy.type = converted constants.%CppHoldsCarbonHoldsNonTrivialDestructor, %Destroy.facet.loc19_22.2 [concrete = constants.%Destroy.facet.dd9]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.dd9) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_with_virtual_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %VirtualFunction: type = class_type @VirtualFunction [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %VirtualFunction.F.type: type = fn_type @VirtualFunction.F [concrete]
// CHECK:STDOUT:   %VirtualFunction.F: %VirtualFunction.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ebc: %SubobjectDestroy.type = facet_value %VirtualFunction, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ebc) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %VirtualFunction, (%custom_witness.b80, %DestroyImpl.impl_witness.1e6) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %VirtualFunction.F.decl: %VirtualFunction.F.type = fn_decl @VirtualFunction.F [concrete = constants.%VirtualFunction.F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualFunction.vtable {
// CHECK:STDOUT:   imports.%VirtualFunction.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestVirtualFunction(%x.param: %VirtualFunction) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %VirtualFunction = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%VirtualFunction, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.1e6) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%VirtualFunction, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%VirtualFunction, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.1e6) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%VirtualFunction, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %VirtualFunction) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_class_with_virtual_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %VirtualDestructor: type = class_type @VirtualDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %VirtualDestructor.cpp_destructor.type: type = fn_type @VirtualDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %VirtualDestructor.cpp_destructor: %VirtualDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%VirtualDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0a0: %SubobjectDestroy.type = facet_value %VirtualDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.05b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0a0) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %VirtualDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.05b) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualDestructor.vtable {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestVirtualDestructor(%x.param: %VirtualDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %VirtualDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%VirtualDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.05b) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%VirtualDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%VirtualDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.05b) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%VirtualDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

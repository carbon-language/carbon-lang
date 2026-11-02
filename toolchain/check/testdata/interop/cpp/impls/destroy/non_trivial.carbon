// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/non_trivial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/non_trivial.carbon


// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a value's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// --- cpp_types.h

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

template <typename T>
struct ParameterizedDataMember {
  T t;
};

// --- todo_fail_cpp_class_non_trivial_destructor.carbon

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

// --- todo_fail_cpp_class_template.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestHolderOfNonTrivialDestructor(x: Cpp.ParameterizedDataMember(Cpp.NonTrivialDestructor))  {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_cpp_class_types.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

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

// --- todo_fail_carbon_class_with_cpp_non_trivially_destructible_member.carbon

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

// --- todo_fail_carbon_class_derived_from_cpp_non_trivially_destructible.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

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

// --- todo_fail_generic_class_with_cpp_non_trivially_destructible_as_specific_field.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

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

// --- fail_todo_destroy_protected_base_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
class ProtectedDestructor {
protected:
  ~ProtectedDestructor();
};
''';

class Derived {
  extend base: Cpp.ProtectedDestructor;
}

fn TestDerivedFromProtectedDestructor(x: Derived) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE+5]]:3-22: error: cannot access protected member `<C++ destructor>` of type `Cpp.ProtectedDestructor` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE-11]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: imported from: fail_todo_destroy_protected_base_destructor.carbon:[[@LINE-16]]:1 [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_cpp_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

inline Cpp '''c++
using Array = NonTrivialDestructor[2];
''';

fn TestCppArray(x: Cpp.NonTrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- todo_fail_carbon_array.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";


fn TestCarbonArray(x: array(Cpp.NonTrivialDestructor, 2)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);

  // TODO: add CheckImplsDynamicDestroy(x);
}

// CHECK:STDOUT: --- todo_fail_cpp_class_non_trivial_destructor.carbon
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
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.579) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_cpp_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember: type = class_type @ParameterizedDataMember [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.type: type = fn_type @ParameterizedDataMember.cpp_destructor [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor: %ParameterizedDataMember.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%ParameterizedDataMember.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9ae: %SubobjectDestroy.type = facet_value %ParameterizedDataMember, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a74: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9ae) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ParameterizedDataMember, (%custom_witness.fe1, %DestroyImpl.impl_witness.a74) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHolderOfNonTrivialDestructor(%x.param: %ParameterizedDataMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ParameterizedDataMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%ParameterizedDataMember, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.a74) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%ParameterizedDataMember, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%ParameterizedDataMember, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.a74) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%ParameterizedDataMember, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_cpp_class_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember: type = class_type @ParameterizedDataMember [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor.type: type = fn_type @ParameterizedDataMember.cpp_destructor [concrete]
// CHECK:STDOUT:   %ParameterizedDataMember.cpp_destructor: %ParameterizedDataMember.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%ParameterizedDataMember.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ba5: %SubobjectDestroy.type = facet_value %ParameterizedDataMember, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e68: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ba5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e58: %Destroy.type = facet_value %ParameterizedDataMember, (%DestroyImpl.impl_witness.e68, %custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e58) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHolderOfCarbonNonTrivial(%x.param: %ParameterizedDataMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %ParameterizedDataMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc15_22.1: %Destroy.type = facet_value constants.%ParameterizedDataMember, (constants.%DestroyImpl.impl_witness.e68, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.e58]
// CHECK:STDOUT:   %.loc15_22.1: %Destroy.type = converted constants.%ParameterizedDataMember, %Destroy.facet.loc15_22.1 [concrete = constants.%Destroy.facet.e58]
// CHECK:STDOUT:   %Destroy.facet.loc15_22.2: %Destroy.type = facet_value constants.%ParameterizedDataMember, (constants.%DestroyImpl.impl_witness.e68, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.e58]
// CHECK:STDOUT:   %.loc15_22.2: %Destroy.type = converted constants.%ParameterizedDataMember, %Destroy.facet.loc15_22.2 [concrete = constants.%Destroy.facet.e58]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e58) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_carbon_class_with_cpp_non_trivially_destructible_member.carbon
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
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.245) [concrete]
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
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.245) [concrete = constants.%CheckImplsDestroy.specific_fn]
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
// CHECK:STDOUT: --- todo_fail_carbon_class_derived_from_cpp_non_trivially_destructible.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialDestructor: type = class_type @DerivedFromNonTrivialDestructor [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialDestructor.elem: type = unbound_element_type %DerivedFromNonTrivialDestructor, %NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.e72: type = struct_type {.base: %NonTrivialDestructor} [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.66d: %SubobjectDestroy.type = facet_value %DerivedFromNonTrivialDestructor, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7ab: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.66d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a7d: %Destroy.type = facet_value %DerivedFromNonTrivialDestructor, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.7ab) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.a7d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromNonTrivialDestructor(%x.param: %DerivedFromNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromNonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%DerivedFromNonTrivialDestructor, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%DerivedFromNonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%DerivedFromNonTrivialDestructor, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%DerivedFromNonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.a7d]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.a7d) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %struct_type.base.e72) {
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %DerivedFromNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %DerivedFromNonTrivialDestructor.elem = name_ref base, @DerivedFromNonTrivialDestructor.%.loc8 [concrete = @DerivedFromNonTrivialDestructor.%.loc8]
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
// CHECK:STDOUT: --- todo_fail_generic_class_with_cpp_non_trivially_destructible_as_specific_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor.type: type = fn_type @NonTrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor: %NonTrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet.555: %Destroy.type = facet_value %NonTrivialDestructor, (%DestroyImpl.impl_witness.579, %custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %GenericField.7ee: type = class_type @GenericField, @GenericField(%Destroy.facet.555) [concrete]
// CHECK:STDOUT:   %GenericField.elem.6bc: type = unbound_element_type %GenericField.7ee, %NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.t.7ae: type = struct_type {.t: %NonTrivialDestructor} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.eae: %SubobjectDestroy.type = facet_value %GenericField.7ee, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bd5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.eae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3f5: %Destroy.type = facet_value %GenericField.7ee, (%DestroyImpl.impl_witness.bd5, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.3f5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericField(%x.param: %GenericField.7ee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %GenericField.7ee = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%GenericField.7ee, (constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.3f5]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%GenericField.7ee, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.3f5]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%GenericField.7ee, (constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.3f5]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%GenericField.7ee, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.3f5]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.3f5) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %struct_type.t.7ae) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%DestroyImpl.impl_witness.579, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%DestroyImpl.impl_witness.579, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %GenericField.7ee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %GenericField.elem.6bc = name_ref t, @GenericField.%field_decl [concrete = @GenericField.%field_decl]
// CHECK:STDOUT:   %.loc13_22.1: ref %NonTrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%DestroyImpl.impl_witness.579, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%DestroyImpl.impl_witness.579, constants.%custom_witness.fe1) [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.555]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
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
// CHECK:STDOUT: --- fail_todo_destroy_protected_base_destructor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ProtectedDestructor: type = class_type @ProtectedDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %ProtectedDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.454: type = struct_type {.base: %ProtectedDestructor} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.cpp_destructor.type: type = fn_type @ProtectedDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedDestructor.cpp_destructor: %ProtectedDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%ProtectedDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e5d: %SubobjectDestroy.type = facet_value %ProtectedDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.721: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e5d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9c: %Destroy.type = facet_value %ProtectedDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.721) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1fc: %SubobjectDestroy.type = facet_value %Derived, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.660: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f2a: %Destroy.type = facet_value %Derived, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.660) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.f2a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromProtectedDestructor(%x.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Derived = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc23_22.1: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc23_22.1: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc23_22.1 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %Destroy.facet.loc23_22.2: %Destroy.type = facet_value constants.%Derived, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.660) [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %.loc23_22.2: %Destroy.type = converted constants.%Derived, %Destroy.facet.loc23_22.2 [concrete = constants.%Destroy.facet.f2a]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.f2a) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.1(%self.param: ref %struct_type.base.454) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_22.1: ref %ProtectedDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc23_22.1: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.721) [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %.loc23_22.2: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc23_22.1 [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %Destroy.facet.loc23_22.2: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.721) [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %.loc23_22.3: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc23_22.2 [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc23_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_22.2(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %Derived.elem = name_ref base, @Derived.%.loc13 [concrete = @Derived.%.loc13]
// CHECK:STDOUT:   %.loc23_22.1: ref %ProtectedDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc23_22.1: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.721) [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %.loc23_22.2: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc23_22.1 [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %Destroy.facet.loc23_22.2: %Destroy.type = facet_value constants.%ProtectedDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.721) [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   %.loc23_22.3: %Destroy.type = converted constants.%ProtectedDestructor, %Destroy.facet.loc23_22.2 [concrete = constants.%Destroy.facet.c9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc23_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_cpp_array.carbon
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
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.579) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCppArray(%x.param: %NonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %NonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_carbon_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.1d1: type = class_type @Array, @Array(%NonTrivialDestructor, %int_2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor.type: type = fn_type @NonTrivialDestructor.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor.cpp_destructor: %NonTrivialDestructor.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe1: <witness> = custom_witness (%NonTrivialDestructor.cpp_destructor), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e70: %SubobjectDestroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.579: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e70) [concrete]
// CHECK:STDOUT:   %Destroy.facet.9f1: %Destroy.type = facet_value %NonTrivialDestructor, (%custom_witness.fe1, %DestroyImpl.impl_witness.579) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.aa9: %SubobjectDestroy.type = facet_value %Array.1d1, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8b9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.aa9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f44: %Destroy.type = facet_value %Array.1d1, (%custom_witness.b80, %DestroyImpl.impl_witness.8b9) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.f44) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCarbonArray(%x.param: %Array.1d1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.1d1 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%Array.1d1, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.8b9) [concrete = constants.%Destroy.facet.f44]
// CHECK:STDOUT:   %.loc10_22.1: %Destroy.type = converted constants.%Array.1d1, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.f44]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%Array.1d1, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.8b9) [concrete = constants.%Destroy.facet.f44]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%Array.1d1, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.f44]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.f44) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %Array.1d1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_22.1: ref %NonTrivialDestructor = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc10_22.1: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc10_22.2: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc10_22.1 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.2: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc10_22.3: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc10_22.2 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_22.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_22.4: ref %NonTrivialDestructor = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc10_22.3: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc10_22.5: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc10_22.3 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %Destroy.facet.loc10_22.4: %Destroy.type = facet_value constants.%NonTrivialDestructor, (constants.%custom_witness.fe1, constants.%DestroyImpl.impl_witness.579) [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   %.loc10_22.6: %Destroy.type = converted constants.%NonTrivialDestructor, %Destroy.facet.loc10_22.4 [concrete = constants.%Destroy.facet.9f1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

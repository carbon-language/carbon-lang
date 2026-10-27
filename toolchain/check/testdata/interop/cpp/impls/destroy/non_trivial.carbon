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

// --- fail_carbon_class_types.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class HasNonTrivialDestructorMember {
  // CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:10-12: error: name `Cpp` not found [NameNotFound]
  // CHECK:STDERR:
  var n: Cpp.NonTrivialDestructor;
}

// CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:38-63: error: name `NonTrivialDestructorMember` not found [NameNotFound]
// CHECK:STDERR:
fn TestNonTrivialDestructorMember(x: NonTrivialDestructorMember) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:3-19: error: name `CheckImplsDestroy` not found [NameNotFound]
  // CHECK:STDERR:
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

class DerivedFromNonTrivialDestructor {
  // CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:16-18: error: name `Cpp` not found [NameNotFound]
  // CHECK:STDERR:
  extend base: Cpp.NonTrivialDestructor;
}

fn TestDerivedFromNonTrivialDestructor(x: DerivedFromNonTrivialDestructor) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:3-19: error: name `CheckImplsDestroy` not found [NameNotFound]
  // CHECK:STDERR:
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

class GenericField(T: Core.Destroy) {
  var t: T;
}

// CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:37-39: error: name `Cpp` not found [NameNotFound]
// CHECK:STDERR:
fn TestGenericField(x: GenericField(Cpp.NonTrivialDestructor))  {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_carbon_class_types.carbon:[[@LINE+2]]:3-19: error: name `CheckImplsDestroy` not found [NameNotFound]
  // CHECK:STDERR:
  CheckImplsDestroy(x);
  //@dump-sem-ir-end

  // TODO: add CheckImplsTrivialDestroy(x);
  // TODO: add CheckImplsDynamicDestroy(x);
}

// --- fail_bidirectional_interop.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class HasNonTrivialDestructorMember {
  var n: Cpp.NonTrivialDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsNonTrivialDestructor {
  // CHECK:STDERR: fail_bidirectional_interop.carbon:[[@LINE+4]]:3-10: error: no type named 'CarbonHoldsNonTrivialDestructor' in namespace 'Carbon'; did you mean 'CppHoldsCarbonHoldsNonTrivialDestructor'? [CppInteropParseError]
  // CHECK:STDERR: fail_bidirectional_interop.carbon:[[@LINE+3]]:3-41: note: replace with `CppHoldsCarbonHoldsNonTrivialDestructor` [CppInteropFixItReplacement]
  // CHECK:STDERR: fail_bidirectional_interop.carbon:[[@LINE-3]]:8: note: 'CppHoldsCarbonHoldsNonTrivialDestructor' declared here [CppInteropParseNote]
  // CHECK:STDERR:
  Carbon::HasNonTrivialDestructorMember c;
};
''';

fn TestBidirectionalInterop(x: Cpp.CppHoldsCarbonHoldsNonTrivialDestructor) {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_bidirectional_interop.carbon:[[@LINE+2]]:3-19: error: name `CheckImplsDestroy` not found [NameNotFound]
  // CHECK:STDERR:
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
// CHECK:STDOUT: --- fail_carbon_class_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %DerivedFromNonTrivialDestructor: type = class_type @DerivedFromNonTrivialDestructor [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNonTrivialDestructorMember(%x.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: <error> = name_ref CheckImplsDestroy, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, %x [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromNonTrivialDestructor(%x.param: %DerivedFromNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: <error> = name_ref CheckImplsDestroy, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref: %DerivedFromNonTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericField(%x.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: <error> = name_ref CheckImplsDestroy, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, %x [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bidirectional_interop.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %CppHoldsCarbonHoldsNonTrivialDestructor: type = class_type @CppHoldsCarbonHoldsNonTrivialDestructor [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBidirectionalInterop(%x.param: %CppHoldsCarbonHoldsNonTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: <error> = name_ref CheckImplsDestroy, <error> [concrete = <error>]
// CHECK:STDOUT:   %x.ref: %CppHoldsCarbonHoldsNonTrivialDestructor = name_ref x, %x
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

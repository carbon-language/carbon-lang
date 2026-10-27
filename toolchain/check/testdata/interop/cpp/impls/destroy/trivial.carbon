// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/trivial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/destroy/trivial.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a vaule's type can be destroyed.
fn CheckImplsDestroy[T: Core.Destroy](_: T) {}

// --- cpp_types.h

struct TrivialDestructor {};

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
};

// --- cpp_function_pointer.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
using FunctionPointer = void(*)();
''';

fn TestFunctionPointer(x: Cpp.FunctionPointer) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

// --- array.carbon

library "[[@TEST_NAME]]";

import Cpp library "cpp_types.h";
import library "check_destroy_format";

fn TestArrayOfTrivialDestructor(x: array(Cpp.TrivialDestructor, 2)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

inline Cpp '''c++
struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};
''';

fn TestEmptyArrayOfNonTrivialDestructor(x: array(Cpp.NonTrivialDestructor, 0)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

fn TestEmptyArrayOfDeletedDestructor(x: array(Cpp.DeletedDestructor, 0)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

// --- todo_fail_cpp_classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestTrivialDestructor(x: Cpp.TrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

inline Cpp '''c++
struct DefaultedDestructor {
  ~DefaultedDestructor() = default;
};
''';

fn TestDefaultedDestructor(x: Cpp.DefaultedDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

inline Cpp '''c++
struct HasTrivialMember {
  TrivialDestructor t;
  int i;
};
''';

fn TestHasTrivialMember(x: Cpp.HasTrivialMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

inline Cpp '''c++
struct DerivedFromTrivial : TrivialDestructor {};
''';

fn TestDerivedFromTrivial(x: Cpp.DerivedFromTrivial) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

inline Cpp '''c++
template <typename T>
struct Holder {
  T t;
};
''';

fn TestHolderOfI32(x: Cpp.Holder(i32)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

// --- carbon_classes.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class TrivialDestructorMember {
  var t: Cpp.TrivialDestructor;
}

fn TestCarbonWithTrivialDestructorMember(x: TrivialDestructorMember) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

class DerivedFromTrivialDestructor {
  extend base: Cpp.TrivialDestructor;
}

fn TestCarbonDerivedFromTrivialDestructor(x: DerivedFromTrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

class GenericClass(T: Core.Destroy) {
  var t: T;
}

fn TestDestroyableBoxOfTrivialDestructor(x: GenericClass(Cpp.TrivialDestructor)) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

// --- todo_fail_trivial_destructor_adatper.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class TrivialDestructorAdapter {
  adapt Cpp.TrivialDestructor;
}

fn TestTrivialDestructorAdapter(x: TrivialDestructorAdapter) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}

// --- nullptr_t.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp inline '''c++
using nullptr_t = decltype(nullptr);
''';

fn TestNullptr(x: Cpp.nullptr_t) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  // TODO: add CheckImplsDynamicDestroy(x)
  //@dump-sem-ir-end
}

// --- fail_todo_class_cpp_carbon_cpp.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

class CarbonHoldsTrivialDestructor {
  var t: Cpp.TrivialDestructor;
}

inline Cpp '''
struct CppHoldsCarbonHoldsTrivialDestructor {
  Carbon::CarbonHoldsTrivialDestructor c;
};
''';

// TODO: This should have trivial destruction, because every subobject does.
// This should be determined by reverse-interop, not Carbon-C++ interop
// because C++ needs to be able to tell that its types are not destructible.
fn TestCppHoldsCarbonHoldsTrivialDestructor(
    x: Cpp.CppHoldsCarbonHoldsTrivialDestructor) {
  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+2]]:3-28: error: name `CheckHasTrivialDestruction` not found [NameNotFound]
  // CHECK:STDERR:
  CheckHasTrivialDestruction(x);
  // CHECK:STDERR: fail_todo_class_cpp_carbon_cpp.carbon:[[@LINE+2]]:3-21: error: name `CheckHasDestruction` not found [NameNotFound]
  // CHECK:STDERR:
  CheckHasDestruction(x);
}

// --- todo_fail_const.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "cpp_types.h";

fn TestConst(x: const Cpp.TrivialDestructor) {
  //@dump-sem-ir-begin
  CheckImplsDestroy(x);
  // TODO: add CheckImplsTrivialDestroy(x)
  //@dump-sem-ir-end

  // TODO: add CheckImplsDynamicDestroy(x)
}


// CHECK:STDOUT: --- cpp_function_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type61000000 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.659, (%custom_witness.b80, %DestroyImpl.impl_witness.381) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestFunctionPointer(%x.param: %.659) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %.659 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc11_22.1: %Destroy.type = facet_value constants.%.659, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.381) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_22.1: %Destroy.type = converted constants.%.659, %Destroy.facet.loc11_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc11_22.2: %Destroy.type = facet_value constants.%.659, (constants.%custom_witness.b80, constants.%DestroyImpl.impl_witness.381) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_22.2: %Destroy.type = converted constants.%.659, %Destroy.facet.loc11_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Array.48c: type = class_type @Array, @Array(%TrivialDestructor, %int_2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fda: %SubobjectDestroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bf2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fda) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0af: %Destroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.bf2) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d3f: %SubobjectDestroy.type = facet_value %Array.48c, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.106: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d3f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bef: %Destroy.type = facet_value %Array.48c, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.106) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.94e: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.bef) [concrete]
// CHECK:STDOUT:   %NonTrivialDestructor: type = class_type @NonTrivialDestructor [concrete]
// CHECK:STDOUT:   %Array.0e6: type = class_type @Array, @Array(%NonTrivialDestructor, %int_0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c9f: %SubobjectDestroy.type = facet_value %Array.0e6, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fe1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c9f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.225: %Destroy.type = facet_value %Array.0e6, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.fe1) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.2c2: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.225) [concrete]
// CHECK:STDOUT:   %DeletedDestructor: type = class_type @DeletedDestructor [concrete]
// CHECK:STDOUT:   %Array.7de: type = class_type @Array, @Array(%DeletedDestructor, %int_0) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.177: %SubobjectDestroy.type = facet_value %Array.7de, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.364: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.177) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5e7: %Destroy.type = facet_value %Array.7de, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.364) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.3da: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.5e7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestArrayOfTrivialDestructor(%x.param: %Array.48c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.48c = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%Array.48c, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.106) [concrete = constants.%Destroy.facet.bef]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%Array.48c, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.bef]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%Array.48c, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.106) [concrete = constants.%Destroy.facet.bef]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%Array.48c, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.bef]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.bef) [concrete = constants.%CheckImplsDestroy.specific_fn.94e]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %Array.48c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_22.1: ref %TrivialDestructor = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc9_22.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_22.4: ref %TrivialDestructor = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.facet.loc9_22.3: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.5: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.3 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.4: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.6: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.4 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc9_22.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc9_22.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyArrayOfNonTrivialDestructor(%x.param: %Array.0e6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.0e6 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc23_22.1: %Destroy.type = facet_value constants.%Array.0e6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.fe1) [concrete = constants.%Destroy.facet.225]
// CHECK:STDOUT:   %.loc23_22.1: %Destroy.type = converted constants.%Array.0e6, %Destroy.facet.loc23_22.1 [concrete = constants.%Destroy.facet.225]
// CHECK:STDOUT:   %Destroy.facet.loc23_22.2: %Destroy.type = facet_value constants.%Array.0e6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.fe1) [concrete = constants.%Destroy.facet.225]
// CHECK:STDOUT:   %.loc23_22.2: %Destroy.type = converted constants.%Array.0e6, %Destroy.facet.loc23_22.2 [concrete = constants.%Destroy.facet.225]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.225) [concrete = constants.%CheckImplsDestroy.specific_fn.2c2]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23(%self.param: ref %Array.0e6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyArrayOfDeletedDestructor(%x.param: %Array.7de) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Array.7de = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc31_22.1: %Destroy.type = facet_value constants.%Array.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.364) [concrete = constants.%Destroy.facet.5e7]
// CHECK:STDOUT:   %.loc31_22.1: %Destroy.type = converted constants.%Array.7de, %Destroy.facet.loc31_22.1 [concrete = constants.%Destroy.facet.5e7]
// CHECK:STDOUT:   %Destroy.facet.loc31_22.2: %Destroy.type = facet_value constants.%Array.7de, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.364) [concrete = constants.%Destroy.facet.5e7]
// CHECK:STDOUT:   %.loc31_22.2: %Destroy.type = converted constants.%Array.7de, %Destroy.facet.loc31_22.2 [concrete = constants.%Destroy.facet.5e7]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.5e7) [concrete = constants.%CheckImplsDestroy.specific_fn.3da]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31(%self.param: ref %Array.7de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_cpp_classes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fda: %SubobjectDestroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bf2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fda) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0af: %Destroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.bf2) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.9a9: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.0af) [concrete]
// CHECK:STDOUT:   %DefaultedDestructor: type = class_type @DefaultedDestructor [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5fd2: %SubobjectDestroy.type = facet_value %DefaultedDestructor, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0e6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5fd2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.cdb: %Destroy.type = facet_value %DefaultedDestructor, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0e6) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.1a5: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.cdb) [concrete]
// CHECK:STDOUT:   %HasTrivialMember: type = class_type @HasTrivialMember [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc40 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4ac: %SubobjectDestroy.type = facet_value %HasTrivialMember, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.757: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4ac) [concrete]
// CHECK:STDOUT:   %Destroy.facet.304: %Destroy.type = facet_value %HasTrivialMember, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.757) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.9dc: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.304) [concrete]
// CHECK:STDOUT:   %DerivedFromTrivial: type = class_type @DerivedFromTrivial [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc53 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.59f: %SubobjectDestroy.type = facet_value %DerivedFromTrivial, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.732: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.59f) [concrete]
// CHECK:STDOUT:   %Destroy.facet.23a: %Destroy.type = facet_value %DerivedFromTrivial, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.732) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.75d: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.23a) [concrete]
// CHECK:STDOUT:   %Holder: type = class_type @Holder [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc69 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3cc: %SubobjectDestroy.type = facet_value %Holder, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2ae: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3cc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.39f: %Destroy.type = facet_value %Holder, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.2ae) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.74f: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.39f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialDestructor(%x.param: %TrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %TrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.0af) [concrete = constants.%CheckImplsDestroy.specific_fn.9a9]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDefaultedDestructor(%x.param: %DefaultedDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DefaultedDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc24_22.1: %Destroy.type = facet_value constants.%DefaultedDestructor, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0e6) [concrete = constants.%Destroy.facet.cdb]
// CHECK:STDOUT:   %.loc24_22.1: %Destroy.type = converted constants.%DefaultedDestructor, %Destroy.facet.loc24_22.1 [concrete = constants.%Destroy.facet.cdb]
// CHECK:STDOUT:   %Destroy.facet.loc24_22.2: %Destroy.type = facet_value constants.%DefaultedDestructor, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0e6) [concrete = constants.%Destroy.facet.cdb]
// CHECK:STDOUT:   %.loc24_22.2: %Destroy.type = converted constants.%DefaultedDestructor, %Destroy.facet.loc24_22.2 [concrete = constants.%Destroy.facet.cdb]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.cdb) [concrete = constants.%CheckImplsDestroy.specific_fn.1a5]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24(%self.param: ref %DefaultedDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHasTrivialMember(%x.param: %HasTrivialMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %HasTrivialMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc40_22.1: %Destroy.type = facet_value constants.%HasTrivialMember, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.757) [concrete = constants.%Destroy.facet.304]
// CHECK:STDOUT:   %.loc40_22.1: %Destroy.type = converted constants.%HasTrivialMember, %Destroy.facet.loc40_22.1 [concrete = constants.%Destroy.facet.304]
// CHECK:STDOUT:   %Destroy.facet.loc40_22.2: %Destroy.type = facet_value constants.%HasTrivialMember, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.757) [concrete = constants.%Destroy.facet.304]
// CHECK:STDOUT:   %.loc40_22.2: %Destroy.type = converted constants.%HasTrivialMember, %Destroy.facet.loc40_22.2 [concrete = constants.%Destroy.facet.304]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.304) [concrete = constants.%CheckImplsDestroy.specific_fn.9dc]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc40(%self.param: ref %HasTrivialMember) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDerivedFromTrivial(%x.param: %DerivedFromTrivial) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromTrivial = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc53_22.1: %Destroy.type = facet_value constants.%DerivedFromTrivial, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.732) [concrete = constants.%Destroy.facet.23a]
// CHECK:STDOUT:   %.loc53_22.1: %Destroy.type = converted constants.%DerivedFromTrivial, %Destroy.facet.loc53_22.1 [concrete = constants.%Destroy.facet.23a]
// CHECK:STDOUT:   %Destroy.facet.loc53_22.2: %Destroy.type = facet_value constants.%DerivedFromTrivial, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.732) [concrete = constants.%Destroy.facet.23a]
// CHECK:STDOUT:   %.loc53_22.2: %Destroy.type = converted constants.%DerivedFromTrivial, %Destroy.facet.loc53_22.2 [concrete = constants.%Destroy.facet.23a]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.23a) [concrete = constants.%CheckImplsDestroy.specific_fn.75d]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53(%self.param: ref %DerivedFromTrivial) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestHolderOfI32(%x.param: %Holder) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Holder = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc69_22.1: %Destroy.type = facet_value constants.%Holder, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.2ae) [concrete = constants.%Destroy.facet.39f]
// CHECK:STDOUT:   %.loc69_22.1: %Destroy.type = converted constants.%Holder, %Destroy.facet.loc69_22.1 [concrete = constants.%Destroy.facet.39f]
// CHECK:STDOUT:   %Destroy.facet.loc69_22.2: %Destroy.type = facet_value constants.%Holder, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.2ae) [concrete = constants.%Destroy.facet.39f]
// CHECK:STDOUT:   %.loc69_22.2: %Destroy.type = converted constants.%Holder, %Destroy.facet.loc69_22.2 [concrete = constants.%Destroy.facet.39f]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.39f) [concrete = constants.%CheckImplsDestroy.specific_fn.74f]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc69(%self.param: ref %Holder) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- carbon_classes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestructorMember: type = class_type @TrivialDestructorMember [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TrivialDestructorMember.elem: type = unbound_element_type %TrivialDestructorMember, %TrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.t.741: type = struct_type {.t: %TrivialDestructor} [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.fda: %SubobjectDestroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bf2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.fda) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0af: %Destroy.type = facet_value %TrivialDestructor, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.bf2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.92b: %SubobjectDestroy.type = facet_value %TrivialDestructorMember, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ab5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.92b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f90: %Destroy.type = facet_value %TrivialDestructorMember, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.ab5) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.1a4: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.f90) [concrete]
// CHECK:STDOUT:   %DerivedFromTrivialDestructor: type = class_type @DerivedFromTrivialDestructor [concrete]
// CHECK:STDOUT:   %DerivedFromTrivialDestructor.elem: type = unbound_element_type %DerivedFromTrivialDestructor, %TrivialDestructor [concrete]
// CHECK:STDOUT:   %struct_type.base.7ca: type = struct_type {.base: %TrivialDestructor} [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc25_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.68b: %SubobjectDestroy.type = facet_value %DerivedFromTrivialDestructor, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b53: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.68b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.49e: %Destroy.type = facet_value %DerivedFromTrivialDestructor, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.b53) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.68c: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.49e) [concrete]
// CHECK:STDOUT:   %GenericClass.a76: type = class_type @GenericClass, @GenericClass(%Destroy.facet.0af) [concrete]
// CHECK:STDOUT:   %GenericClass.elem.88b: type = unbound_element_type %GenericClass.a76, %TrivialDestructor [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc37 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.66a: %SubobjectDestroy.type = facet_value %GenericClass.a76, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0f9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.66a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.560: %Destroy.type = facet_value %GenericClass.a76, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.0f9) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn.e2c: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.560) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCarbonWithTrivialDestructorMember(%x.param: %TrivialDestructorMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %TrivialDestructorMember = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%TrivialDestructorMember, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.ab5) [concrete = constants.%Destroy.facet.f90]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%TrivialDestructorMember, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.f90]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%TrivialDestructorMember, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.ab5) [concrete = constants.%Destroy.facet.f90]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%TrivialDestructorMember, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.f90]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.f90) [concrete = constants.%CheckImplsDestroy.specific_fn.1a4]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %struct_type.t.741) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_22.1: ref %TrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.3(%self.param: ref %TrivialDestructorMember) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %TrivialDestructorMember.elem = name_ref t, @TrivialDestructorMember.%field_decl [concrete = @TrivialDestructorMember.%field_decl]
// CHECK:STDOUT:   %.loc13_22.1: ref %TrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc13_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestCarbonDerivedFromTrivialDestructor(%x.param: %DerivedFromTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %DerivedFromTrivialDestructor = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc25_22.1: %Destroy.type = facet_value constants.%DerivedFromTrivialDestructor, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.b53) [concrete = constants.%Destroy.facet.49e]
// CHECK:STDOUT:   %.loc25_22.1: %Destroy.type = converted constants.%DerivedFromTrivialDestructor, %Destroy.facet.loc25_22.1 [concrete = constants.%Destroy.facet.49e]
// CHECK:STDOUT:   %Destroy.facet.loc25_22.2: %Destroy.type = facet_value constants.%DerivedFromTrivialDestructor, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.b53) [concrete = constants.%Destroy.facet.49e]
// CHECK:STDOUT:   %.loc25_22.2: %Destroy.type = converted constants.%DerivedFromTrivialDestructor, %Destroy.facet.loc25_22.2 [concrete = constants.%Destroy.facet.49e]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.49e) [concrete = constants.%CheckImplsDestroy.specific_fn.68c]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_22.1(%self.param: ref %struct_type.base.7ca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_22.1: ref %TrivialDestructor = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc25_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc25_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc25_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc25_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc25_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc25_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc25_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_22.2(%self.param: ref %DerivedFromTrivialDestructor) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %base.ref: %DerivedFromTrivialDestructor.elem = name_ref base, @DerivedFromTrivialDestructor.%.loc20 [concrete = @DerivedFromTrivialDestructor.%.loc20]
// CHECK:STDOUT:   %.loc25_22.1: ref %TrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc25_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc25_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc25_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc25_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc25_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc25_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc25_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDestroyableBoxOfTrivialDestructor(%x.param: %GenericClass.a76) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %GenericClass.a76 = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc37_22.1: %Destroy.type = facet_value constants.%GenericClass.a76, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0f9) [concrete = constants.%Destroy.facet.560]
// CHECK:STDOUT:   %.loc37_22.1: %Destroy.type = converted constants.%GenericClass.a76, %Destroy.facet.loc37_22.1 [concrete = constants.%Destroy.facet.560]
// CHECK:STDOUT:   %Destroy.facet.loc37_22.2: %Destroy.type = facet_value constants.%GenericClass.a76, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.0f9) [concrete = constants.%Destroy.facet.560]
// CHECK:STDOUT:   %.loc37_22.2: %Destroy.type = converted constants.%GenericClass.a76, %Destroy.facet.loc37_22.2 [concrete = constants.%Destroy.facet.560]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.560) [concrete = constants.%CheckImplsDestroy.specific_fn.e2c]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc37(%self.param: ref %GenericClass.a76) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: %GenericClass.elem.88b = name_ref t, @GenericClass.%field_decl [concrete = @GenericClass.%field_decl]
// CHECK:STDOUT:   %.loc37_22.1: ref %TrivialDestructor = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc37_22.1: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc37_22.2: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc37_22.1 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %Destroy.facet.loc37_22.2: %Destroy.type = facet_value constants.%TrivialDestructor, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.bf2) [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   %.loc37_22.3: %Destroy.type = converted constants.%TrivialDestructor, %Destroy.facet.loc37_22.2 [concrete = constants.%Destroy.facet.0af]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc37_22.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_trivial_destructor_adatper.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestructorAdapter: type = class_type @TrivialDestructorAdapter [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8b7: %SubobjectDestroy.type = facet_value %TrivialDestructorAdapter, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.be0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8b7) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %TrivialDestructorAdapter, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.be0) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTrivialDestructorAdapter(%x.param: %TrivialDestructorAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %TrivialDestructorAdapter = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc13_22.1: %Destroy.type = facet_value constants.%TrivialDestructorAdapter, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.be0) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.1: %Destroy.type = converted constants.%TrivialDestructorAdapter, %Destroy.facet.loc13_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_22.2: %Destroy.type = facet_value constants.%TrivialDestructorAdapter, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.be0) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_22.2: %Destroy.type = converted constants.%TrivialDestructorAdapter, %Destroy.facet.loc13_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.1(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_22.2(%self.param: ref %TrivialDestructorAdapter) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullptr_t.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %Cpp.void: type = class_type @VoidBase [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Cpp.void [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a43: %SubobjectDestroy.type = facet_value %ptr, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e97: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a43) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.facet.1c1: %Destroy.type = facet_value %ptr, (%DestroyImpl.impl_witness.e97, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.74d: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.1c1) [concrete]
// CHECK:STDOUT:   %.a7d: type = maybe_unformed_type %ptr [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_22.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e68: %SubobjectDestroy.type = facet_value %Cpp.nullptr_t, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.452: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e68) [concrete]
// CHECK:STDOUT:   %Destroy.facet.e32: %Destroy.type = facet_value %Cpp.nullptr_t, (%DestroyImpl.impl_witness.452, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet.e32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.98d = import_ref Core//prelude/types/cpp/nullptr, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.98d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNullptr(%x.param: %Cpp.nullptr_t) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %Cpp.nullptr_t = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc11_22.1: %Destroy.type = facet_value constants.%Cpp.nullptr_t, (constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.e32]
// CHECK:STDOUT:   %.loc11_22.1: %Destroy.type = converted constants.%Cpp.nullptr_t, %Destroy.facet.loc11_22.1 [concrete = constants.%Destroy.facet.e32]
// CHECK:STDOUT:   %Destroy.facet.loc11_22.2: %Destroy.type = facet_value constants.%Cpp.nullptr_t, (constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.e32]
// CHECK:STDOUT:   %.loc11_22.2: %Destroy.type = converted constants.%Cpp.nullptr_t, %Destroy.facet.loc11_22.2 [concrete = constants.%Destroy.facet.e32]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet.e32) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_22.1(%self.param: ref %.a7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_22.2(%self.param: ref %MaybeUnformed.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_22.3(%self.param: ref %Cpp.nullptr_t) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestructor: type = class_type @TrivialDestructor [concrete]
// CHECK:STDOUT:   %const: type = const_type %TrivialDestructor [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.type: type = fn_type @CheckImplsDestroy [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy: %CheckImplsDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_22.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.07a: %SubobjectDestroy.type = facet_value %const, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.aaa: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.07a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %const, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.aaa) [concrete]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy, @CheckImplsDestroy(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckImplsDestroy: %CheckImplsDestroy.type = import_ref Main//check_destroy_format, CheckImplsDestroy, loaded [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %Core.import_ref.9fa = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.9fa), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestConst(%x.param: %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckImplsDestroy.ref: %CheckImplsDestroy.type = name_ref CheckImplsDestroy, imports.%Main.CheckImplsDestroy [concrete = constants.%CheckImplsDestroy]
// CHECK:STDOUT:   %x.ref: %const = name_ref x, %x
// CHECK:STDOUT:   %Destroy.facet.loc9_22.1: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.aaa) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.1: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_22.2: %Destroy.type = facet_value constants.%const, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.aaa) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_22.2: %Destroy.type = converted constants.%const, %Destroy.facet.loc9_22.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %CheckImplsDestroy.specific_fn: <specific function> = specific_function %CheckImplsDestroy.ref, @CheckImplsDestroy(constants.%Destroy.facet) [concrete = constants.%CheckImplsDestroy.specific_fn]
// CHECK:STDOUT:   %CheckImplsDestroy.call: init %empty_tuple.type = call %CheckImplsDestroy.specific_fn(%x.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.1(%self.param: ref %TrivialDestructor) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_22.2(%self.param: ref %const) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

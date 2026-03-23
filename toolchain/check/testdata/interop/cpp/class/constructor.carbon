// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C();
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.C = Cpp.C.C();
  //@dump-sem-ir-end
}

// ============================================================================
// Non default constructor
// ============================================================================

// --- non_default.h

class C {
 public:
  C(int x, int y);
};

// --- import_non_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.C = Cpp.C.C(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Multiple constructors
// ============================================================================

// --- multiple.h

class C {
 public:
  C();
  C(int x, int y);
};

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C();
  let unused c2: Cpp.C = Cpp.C.C(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor with default arguments
// ============================================================================

// --- default_arguments.h

class C {
 public:
  C(int x, int y = 8);
};

// --- import_default_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "default_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8, 9);
  let unused c2: Cpp.C = Cpp.C.C(8);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor templates
// ============================================================================

// --- template.h

class C {
 public:
  C(int);
  template<typename T> C(T);
};

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.C = Cpp.C.C(123);
  var unused c2: Cpp.C = Cpp.C.C(true);
  var unused c3: Cpp.C = Cpp.C.C(&c1);
  //@dump-sem-ir-end
}

// ============================================================================
// Deleted constructor
// ============================================================================

// --- deleted.h

class C {
 public:
  C() = delete;
};

class D {
 public:
  D(int) = delete;
  D(short);
};

// --- fail_import_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "deleted.h";

fn F() {
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+8]]:33: error: call to deleted function 'C' [CppInteropParseError]
  // CHECK:STDERR:   166 |   let unused c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:       |                                 ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:142:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:   142 |   C() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let unused c: Cpp.C = Cpp.C.C();

  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+12]]:34: error: call to deleted function 'D' [CppInteropParseError]
  // CHECK:STDERR:   180 |   let unused d: Cpp.D = Cpp.D.D(0);
  // CHECK:STDERR:       |                                  ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:147:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:   147 |   D(int) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:148:3: note: candidate constructor [CppInteropParseNote]
  // CHECK:STDERR:   148 |   D(short);
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let unused d: Cpp.D = Cpp.D.D(0);
}

// ============================================================================
// Implicit single argument constructor
// ============================================================================

// --- implicit_single_argument.h

class C {
 public:
  C(int x);
};

// --- import_implicit_single_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_single_argument.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8);
  let unused c2: Cpp.C = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Implicit multi arguments constructor
// ============================================================================

// --- implicit_multi_arguments.h

class C {
 public:
  C(int x, int y = 8);
};

// --- import_implicit_multi_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_multi_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.C = Cpp.C.C(8, 9);
  let unused c2: Cpp.C = Cpp.C.C(8);
  let unused c3: Cpp.C = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Trying to call a constructor of an incomplete class
// ============================================================================

// --- incomplete.h

class C;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
   // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+8]]:4: error: member access into incomplete class `Cpp.C` [QualifiedExprInIncompleteClassScope]
   // CHECK:STDERR:    Cpp.C.C();
   // CHECK:STDERR:    ^~~~~~~
   // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
   // CHECK:STDERR: ./incomplete.h:238:7: note: class was forward declared here [ClassForwardDeclaredHere]
   // CHECK:STDERR: class C;
   // CHECK:STDERR:       ^
   // CHECK:STDERR:
   Cpp.C.C();
}

// CHECK:STDOUT: --- import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc28.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc28.2: init %C to %.loc28.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28.3: ref %C = temporary %.loc28.1, %.loc28.2
// CHECK:STDOUT:   %.loc28.4: %C = acquire_value %.loc28.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc28.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc28.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc28.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f5c: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.db6: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.266: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.f41: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.217 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc28: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28.1: <bound method> = bound_method %int_123, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f5c]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28.2: <bound method> = bound_method %int_123, %specific_fn.loc28 [concrete = constants.%bound_method.db6]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i32 = call %bound_method.loc28.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc28.2: %i32 = converted %int_123, %.loc28.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc30.1: <bound method> = bound_method %int_456, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.266]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30.2: <bound method> = bound_method %int_456, %specific_fn.loc30 [concrete = constants.%bound_method.f41]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc30.2: %i32 = converted %int_456, %.loc30.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc28.2, %.loc30.2, %addr)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc31.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc31.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f5c: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.db6: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.266: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.f41: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.d17: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc28.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc28: %ptr.d9e = addr_of %.loc28.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc28)
// CHECK:STDOUT:   %.loc28.2: init %C to %.loc28.1 = mark_in_place_init %C__carbon_thunk.call.loc28
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28.3: ref %C = temporary %.loc28.1, %.loc28.2
// CHECK:STDOUT:   %.loc28.4: %C = acquire_value %.loc28.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc28.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc39: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc41: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc47.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc44: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.1: <bound method> = bound_method %int_123, %impl.elem0.loc44 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f5c]
// CHECK:STDOUT:   %specific_fn.loc44: <specific function> = specific_function %impl.elem0.loc44, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.2: <bound method> = bound_method %int_123, %specific_fn.loc44 [concrete = constants.%bound_method.db6]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44: init %i32 = call %bound_method.loc44.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc44.2: %i32 = converted %int_123, %.loc44.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %impl.elem0.loc46: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc46.1: <bound method> = bound_method %int_456, %impl.elem0.loc46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.266]
// CHECK:STDOUT:   %specific_fn.loc46: <specific function> = specific_function %impl.elem0.loc46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46.2: <bound method> = bound_method %int_456, %specific_fn.loc46 [concrete = constants.%bound_method.f41]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46: init %i32 = call %bound_method.loc46.2(%int_456) [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc46.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %.loc46.2: %i32 = converted %int_456, %.loc46.1 [concrete = constants.%int_456.d17]
// CHECK:STDOUT:   %addr.loc47: %ptr.d9e = addr_of %.loc47.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc47: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc44.2, %.loc46.2, %addr.loc47)
// CHECK:STDOUT:   %.loc47.2: init %C to %.loc47.1 = mark_in_place_init %C__carbon_thunk.call.loc47
// CHECK:STDOUT:   %.loc35: type = splice_block %C.ref.loc35 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc35: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc47.3: ref %C = temporary %.loc47.1, %.loc47.2
// CHECK:STDOUT:   %.loc47.4: %C = acquire_value %.loc47.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc47.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc47: <bound method> = bound_method %.loc47.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc47: init %empty_tuple.type = call %C.cpp_destructor.bound.loc47(%.loc47.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc28: <bound method> = bound_method %.loc28.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc28: init %empty_tuple.type = call %C.cpp_destructor.bound.loc28(%.loc28.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_default_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e07: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.661: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc28: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc28: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28.1: <bound method> = bound_method %int_8.loc28, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28.2: <bound method> = bound_method %int_8.loc28, %specific_fn.loc28 [concrete = constants.%bound_method.e07]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i32 = call %bound_method.loc28.2(%int_8.loc28) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.2: %i32 = converted %int_8.loc28, %.loc28.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc30.1: <bound method> = bound_method %int_9, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30.2: <bound method> = bound_method %int_9, %specific_fn.loc30 [concrete = constants.%bound_method.661]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc30.2: %i32 = converted %int_9, %.loc30.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc31: %ptr.d9e = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc28.2, %.loc30.2, %addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call.loc31
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc42: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc44: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc47: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc48.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc47: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %int_8.loc47, %impl.elem0.loc47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %int_8.loc47, %specific_fn.loc47 [concrete = constants.%bound_method.e07]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47: init %i32 = call %bound_method.loc47.2(%int_8.loc47) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc47.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc47.2: %i32 = converted %int_8.loc47, %.loc47.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc48: %ptr.d9e = addr_of %.loc48.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc48: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc47.2, %addr.loc48)
// CHECK:STDOUT:   %.loc48.2: init %C to %.loc48.1 = mark_in_place_init %C__carbon_thunk.call.loc48
// CHECK:STDOUT:   %.loc38: type = splice_block %C.ref.loc38 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc38: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc48.3: ref %C = temporary %.loc48.1, %.loc48.2
// CHECK:STDOUT:   %.loc48.4: %C = acquire_value %.loc48.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc48.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc48: <bound method> = bound_method %.loc48.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc48: init %empty_tuple.type = call %C.cpp_destructor.bound.loc48(%.loc48.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.f7f: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.348: <witness> = impl_witness imports.%Copy.impl_witness_table.3cc [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.348) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6dd: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.86d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6dd, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.3: type = fn_type @C__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.3: %C__carbon_thunk.type.65f120.3 = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.595: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc124, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cc = impl_witness_table (%Core.import_ref.595), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.3: %C__carbon_thunk.type.65f120.3 = fn_decl @C__carbon_thunk.3 [concrete = constants.%C__carbon_thunk.d98342.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.217 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %.loc14: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %impl.elem0.loc27: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc27.1: <bound method> = bound_method %int_123, %impl.elem0.loc27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27.2(%int_123) [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %.loc27.2: %i32 = converted %int_123, %.loc27.1 [concrete = constants.%int_123.f7f]
// CHECK:STDOUT:   %addr.loc28: %ptr.d9e = addr_of %.loc14
// CHECK:STDOUT:   %C__carbon_thunk.call.loc28: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc27.2, %addr.loc28)
// CHECK:STDOUT:   %.loc28: init %C to %.loc14 = mark_in_place_init %C__carbon_thunk.call.loc28
// CHECK:STDOUT:   assign %c1.var, %.loc28
// CHECK:STDOUT:   %.loc18: type = splice_block %C.ref.loc18 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.217 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc39: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc41: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc30: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %.loc44.1: ref bool = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc44: %.86d = impl_witness_access constants.%Copy.impl_witness.348, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc44: <bound method> = bound_method %true, %impl.elem0.loc44 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc44(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc44.2: ref bool = temporary %.loc44.1, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc45.1: %ptr.bb2 = addr_of %.loc44.2
// CHECK:STDOUT:   %addr.loc45.2: %ptr.d9e = addr_of %.loc30
// CHECK:STDOUT:   %C__carbon_thunk.call.loc45: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%addr.loc45.1, %addr.loc45.2)
// CHECK:STDOUT:   %.loc45: init %C to %.loc30 = mark_in_place_init %C__carbon_thunk.call.loc45
// CHECK:STDOUT:   assign %c2.var, %.loc45
// CHECK:STDOUT:   %.loc35: type = splice_block %C.ref.loc35 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc35: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.217 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc56: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc58: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %c1.ref: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %addr.loc61: %ptr.d9e = addr_of %c1.ref
// CHECK:STDOUT:   %.loc47: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc63: %ptr.d9e = addr_of %.loc47
// CHECK:STDOUT:   %C__carbon_thunk.call.loc63: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.3(%addr.loc61, %addr.loc63)
// CHECK:STDOUT:   %.loc63: init %C to %.loc47 = mark_in_place_init %C__carbon_thunk.call.loc63
// CHECK:STDOUT:   assign %c3.var, %.loc63
// CHECK:STDOUT:   %.loc52: type = splice_block %C.ref.loc52 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc51: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc52: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc47: <bound method> = bound_method %c3.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc47: init %empty_tuple.type = call %C.cpp_destructor.bound.loc47(%c3.var)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc44.2)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc30: <bound method> = bound_method %c2.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc30: init %empty_tuple.type = call %C.cpp_destructor.bound.loc30(%c2.var)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc14: <bound method> = bound_method %c1.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc14: init %empty_tuple.type = call %C.cpp_destructor.bound.loc14(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_implicit_single_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e07: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.530: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc197, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc28: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc29.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc28: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28.1: <bound method> = bound_method %int_8.loc28, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28.2: <bound method> = bound_method %int_8.loc28, %specific_fn.loc28 [concrete = constants.%bound_method.e07]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc28.2(%int_8.loc28) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.2: %i32 = converted %int_8.loc28, %.loc28.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc29: %ptr.d9e = addr_of %.loc29.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc29: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc28.2, %addr.loc29)
// CHECK:STDOUT:   %.loc29.2: init %C to %.loc29.1 = mark_in_place_init %C__carbon_thunk.call.loc29
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc29.3: ref %C = temporary %.loc29.1, %.loc29.2
// CHECK:STDOUT:   %.loc29.4: %C = acquire_value %.loc29.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc29.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc39: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc40: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc40.1: <bound method> = bound_method %int_8.loc39, %impl.elem0.loc40 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc40: <specific function> = specific_function %impl.elem0.loc40, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40.2: <bound method> = bound_method %int_8.loc39, %specific_fn.loc40 [concrete = constants.%bound_method.530]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc40.2(%int_8.loc39) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc40.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc40.2: %i32 = converted %int_8.loc39, %.loc40.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc36: type = splice_block %C.ref.loc36 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc36: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc40.3: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc40: %ptr.d9e = addr_of %.loc40.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc40: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc40.2, %addr.loc40)
// CHECK:STDOUT:   %.loc40.4: init %C to %.loc40.3 = mark_in_place_init %C__carbon_thunk.call.loc40
// CHECK:STDOUT:   %.loc40.5: init %C = converted %.loc40.2, %.loc40.4
// CHECK:STDOUT:   %.loc40.6: ref %C = temporary %.loc40.3, %.loc40.5
// CHECK:STDOUT:   %.loc40.7: %C = acquire_value %.loc40.6
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc40.7
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc40: <bound method> = bound_method %.loc40.6, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc40: init %empty_tuple.type = call %C.cpp_destructor.bound.loc40(%.loc40.6)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc29: <bound method> = bound_method %.loc29.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc29: init %empty_tuple.type = call %C.cpp_destructor.bound.loc29(%.loc29.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_implicit_multi_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.217: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e07: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.661: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.530: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc197, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.217 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc25: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc28: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc31.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc28: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28.1: <bound method> = bound_method %int_8.loc28, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28.2: <bound method> = bound_method %int_8.loc28, %specific_fn.loc28 [concrete = constants.%bound_method.e07]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i32 = call %bound_method.loc28.2(%int_8.loc28) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc28.2: %i32 = converted %int_8.loc28, %.loc28.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc30.1: <bound method> = bound_method %int_9, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30.2: <bound method> = bound_method %int_9, %specific_fn.loc30 [concrete = constants.%bound_method.661]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc30.2: %i32 = converted %int_9, %.loc30.1 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %addr.loc31: %ptr.d9e = addr_of %.loc31.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%.loc28.2, %.loc30.2, %addr.loc31)
// CHECK:STDOUT:   %.loc31.2: init %C to %.loc31.1 = mark_in_place_init %C__carbon_thunk.call.loc31
// CHECK:STDOUT:   %.loc19: type = splice_block %C.ref.loc19 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc19: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31.3: ref %C = temporary %.loc31.1, %.loc31.2
// CHECK:STDOUT:   %.loc31.4: %C = acquire_value %.loc31.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc31.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.217 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc42: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc44: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc47: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc48.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc47: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %int_8.loc47, %impl.elem0.loc47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %int_8.loc47, %specific_fn.loc47 [concrete = constants.%bound_method.e07]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47: init %i32 = call %bound_method.loc47.2(%int_8.loc47) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc47.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc47.2: %i32 = converted %int_8.loc47, %.loc47.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %addr.loc48: %ptr.d9e = addr_of %.loc48.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc48: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc47.2, %addr.loc48)
// CHECK:STDOUT:   %.loc48.2: init %C to %.loc48.1 = mark_in_place_init %C__carbon_thunk.call.loc48
// CHECK:STDOUT:   %.loc38: type = splice_block %C.ref.loc38 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc38: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc48.3: ref %C = temporary %.loc48.1, %.loc48.2
// CHECK:STDOUT:   %.loc48.4: %C = acquire_value %.loc48.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc48.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.217 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc58: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc59: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc59.1: <bound method> = bound_method %int_8.loc58, %impl.elem0.loc59 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc59: <specific function> = specific_function %impl.elem0.loc59, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc59.2: <bound method> = bound_method %int_8.loc58, %specific_fn.loc59 [concrete = constants.%bound_method.530]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc59.2(%int_8.loc58) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc59.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc59.2: %i32 = converted %int_8.loc58, %.loc59.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc55: type = splice_block %C.ref.loc55 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc54: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc55: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc59.3: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc59: %ptr.d9e = addr_of %.loc59.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc59: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%.loc59.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59.4: init %C to %.loc59.3 = mark_in_place_init %C__carbon_thunk.call.loc59
// CHECK:STDOUT:   %.loc59.5: init %C = converted %.loc59.2, %.loc59.4
// CHECK:STDOUT:   %.loc59.6: ref %C = temporary %.loc59.3, %.loc59.5
// CHECK:STDOUT:   %.loc59.7: %C = acquire_value %.loc59.6
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc59.7
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc59: <bound method> = bound_method %.loc59.6, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc59: init %empty_tuple.type = call %C.cpp_destructor.bound.loc59(%.loc59.6)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc48: <bound method> = bound_method %.loc48.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc48: init %empty_tuple.type = call %C.cpp_destructor.bound.loc48(%.loc48.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc31: init %empty_tuple.type = call %C.cpp_destructor.bound.loc31(%.loc31.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

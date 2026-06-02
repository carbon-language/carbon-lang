// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C();
};

struct NoDefault {
  NoDefault(int);
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.C = Cpp.C.C();
  //@dump-sem-ir-end
}

// --- implicit_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  var unused c: Cpp.C;
  //@dump-sem-ir-end
}

fn G(T:! Core.Default & Core.Destroy) {
  var unused x: T;
}

fn CallG() {
  G(Cpp.C);
}

// --- todo_fail_no_default_decl.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  // TODO: This should fail, but currently does not because the min_prelude
  // version of UnformedInit is currently implemented by every type.
  var unused x: Cpp.NoDefault;
}

// --- fail_no_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn G(T:! Core.Default & Core.Destroy) {
  var unused x: T;
}

fn CallG() {
  // CHECK:STDERR: fail_no_default.carbon:[[@LINE+7]]:3: error: cannot convert type `NoDefault` into type implementing `Core.Default & Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   G(Cpp.NoDefault);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_no_default.carbon:[[@LINE-8]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G(T:! Core.Default & Core.Destroy) {
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G(Cpp.NoDefault);
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
  // CHECK:STDERR:    15 |   let unused c: Cpp.C = Cpp.C.C();
  // CHECK:STDERR:       |                                 ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:4:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     4 |   C() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR:
  let unused c: Cpp.C = Cpp.C.C();

  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+12]]:34: error: call to deleted function 'D' [CppInteropParseError]
  // CHECK:STDERR:    29 |   let unused d: Cpp.D = Cpp.D.D(0);
  // CHECK:STDERR:       |                                  ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:9:3: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     9 |   D(int) = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE-20]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted.h:10:3: note: candidate constructor [CppInteropParseNote]
  // CHECK:STDERR:    10 |   D(short);
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
   // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
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
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.479 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_28: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_30: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_33.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.3a8 = addr_of %.loc8_33.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_33.2: init %C to %.loc8_33.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_20: type = splice_block %C.ref.loc8_20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_33.3: ref %C = temporary %.loc8_33.1, %.loc8_33.2
// CHECK:STDOUT:   %.loc8_33.4: %C = acquire_value %.loc8_33.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_33.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_33.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_33.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.4ca: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.4ca) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.25a: %T.as_type.as.DefaultOrUnformed.impl.Op.type.47c = struct_value () [symbolic]
// CHECK:STDOUT:   %C.Op.type: type = fn_type @C.Op [concrete]
// CHECK:STDOUT:   %C.Op: %C.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.213: <witness> = custom_witness (%C.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.fc6: %Default.type = facet_value %C, (%custom_witness.213) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.44d: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.b72, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.fc6) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ea5: %DefaultOrUnformed.type = facet_value %C, (%DefaultOrUnformed.impl_witness.44d) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dfa: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.47c) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.25a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.b72 = impl_witness_table (%Core.import_ref.dfa), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.479 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.479 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%C, (constants.%DefaultOrUnformed.impl_witness.44d) [concrete = constants.%DefaultOrUnformed.facet.ea5]
// CHECK:STDOUT:   %.loc8_22.1: %DefaultOrUnformed.type = converted constants.%C, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet.ea5]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc8_22.1 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc8_22.2: type = converted %.loc8_22.1, %as_type [concrete = constants.%C]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %C to %.loc8_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %c.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc8_20: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %c.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%c.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.059: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.692: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.5a3: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.853: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.88a: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.b78: %i32 = int_value 456 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.479 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_28: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_30: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc8_41.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_33: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_33.1: <bound method> = bound_method %int_123, %impl.elem0.loc8_33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.059]
// CHECK:STDOUT:   %specific_fn.loc8_33: <specific function> = specific_function %impl.elem0.loc8_33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_33.2: <bound method> = bound_method %int_123, %specific_fn.loc8_33 [concrete = constants.%bound_method.692]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_33: init %i32 = call %bound_method.loc8_33.2(%int_123) [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc8_33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_33 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc8_33.2: %i32 = converted %int_123, %.loc8_33.1 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %impl.elem0.loc8_38: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_38.1: <bound method> = bound_method %int_456, %impl.elem0.loc8_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.853]
// CHECK:STDOUT:   %specific_fn.loc8_38: <specific function> = specific_function %impl.elem0.loc8_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_38.2: <bound method> = bound_method %int_456, %specific_fn.loc8_38 [concrete = constants.%bound_method.88a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_38: init %i32 = call %bound_method.loc8_38.2(%int_456) [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %.loc8_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_38 [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %.loc8_38.2: %i32 = converted %int_456, %.loc8_38.1 [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %addr: %ptr.3a8 = addr_of %.loc8_41.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_33.2, %.loc8_38.2, %addr)
// CHECK:STDOUT:   %.loc8_41.2: init %C to %.loc8_41.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_20: type = splice_block %C.ref.loc8_20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_41.3: ref %C = temporary %.loc8_41.1, %.loc8_41.2
// CHECK:STDOUT:   %.loc8_41.4: %C = acquire_value %.loc8_41.3
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc8_41.4
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc8_41.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc8_41.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.1: %C__carbon_thunk.type.0be6f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.2: %C__carbon_thunk.type.0be6f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.059: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.692: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.5a3: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.853: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.88a: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.b78: %i32 = int_value 456 [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.1: %C__carbon_thunk.type.0be6f8.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.7c7210.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.2: %C__carbon_thunk.type.0be6f8.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.7c7210.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.479 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_34.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.3a8 = addr_of %.loc8_34.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_34.2: init %C to %.loc8_34.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_34.3: ref %C = temporary %.loc8_34.1, %.loc8_34.2
// CHECK:STDOUT:   %.loc8_34.4: %C = acquire_value %.loc8_34.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_34.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.479 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc9_42.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_123, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.059]
// CHECK:STDOUT:   %specific_fn.loc9_34: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_123, %specific_fn.loc9_34 [concrete = constants.%bound_method.692]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_34: init %i32 = call %bound_method.loc9_34.2(%int_123) [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_34 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_123, %.loc9_34.1 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %impl.elem0.loc9_39: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_39.1: <bound method> = bound_method %int_456, %impl.elem0.loc9_39 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.853]
// CHECK:STDOUT:   %specific_fn.loc9_39: <specific function> = specific_function %impl.elem0.loc9_39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_39.2: <bound method> = bound_method %int_456, %specific_fn.loc9_39 [concrete = constants.%bound_method.88a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_39: init %i32 = call %bound_method.loc9_39.2(%int_456) [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %.loc9_39.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_39 [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %.loc9_39.2: %i32 = converted %int_456, %.loc9_39.1 [concrete = constants.%int_456.b78]
// CHECK:STDOUT:   %addr.loc9: %ptr.3a8 = addr_of %.loc9_42.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.2(%.loc9_34.2, %.loc9_39.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_42.2: init %C to %.loc9_42.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_42.3: ref %C = temporary %.loc9_42.1, %.loc9_42.2
// CHECK:STDOUT:   %.loc9_42.4: %C = acquire_value %.loc9_42.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_42.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_42.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_42.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_34.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_34.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_default_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.1: %C__carbon_thunk.type.0be6f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bd2: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.980: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.299: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.e00: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.e6d: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.2: %C__carbon_thunk.type.0be6f8.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.1: %C__carbon_thunk.type.0be6f8.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.7c7210.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.2: %C__carbon_thunk.type.0be6f8.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.7c7210.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.479 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65]
// CHECK:STDOUT:   %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_34 [concrete = constants.%bound_method.bd2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %impl.elem0.loc8_37: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.299]
// CHECK:STDOUT:   %specific_fn.loc8_37: <specific function> = specific_function %impl.elem0.loc8_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_9, %specific_fn.loc8_37 [concrete = constants.%bound_method.e00]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37: init %i32 = call %bound_method.loc8_37.2(%int_9) [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %.loc8_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37 [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %.loc8_37.2: %i32 = converted %int_9, %.loc8_37.1 [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %addr.loc8: %ptr.3a8 = addr_of %.loc8_38.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.1(%.loc8_34.2, %.loc8_37.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_38.2: init %C to %.loc8_38.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.3: ref %C = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %C = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.479 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.bd2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_8.loc9) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_8.loc9, %.loc9_34.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %addr.loc9: %ptr.3a8 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.2(%.loc9_34.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C to %.loc9_35.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_35.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_38.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.1: %C__carbon_thunk.type.0be6f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.5a3: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.2: %C__carbon_thunk.type.0be6f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.bc7: <witness> = impl_witness imports.%Copy.impl_witness_table.aef [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.bc7) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5dd: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.698: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5dd, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.e3b: ref bool = temporary invalid, %true [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.e3b [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.3: type = fn_type @C__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.3: %C__carbon_thunk.type.0be6f8.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.1: %C__carbon_thunk.type.0be6f8.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.7c7210.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.2: %C__carbon_thunk.type.0be6f8.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.7c7210.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.41e: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.aef = impl_witness_table (%Core.import_ref.41e), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.3: %C__carbon_thunk.type.0be6f8.3 = fn_decl @C__carbon_thunk.3 [concrete = constants.%C__carbon_thunk.7c7210.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.479 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.479 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1.var: ref %C = var %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_22: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_24: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %.loc8_3: ref %C = splice_block %c1.var {}
// CHECK:STDOUT:   %impl.elem0.loc8: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_27.1: <bound method> = bound_method %int_123, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_27.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_27.2(%int_123) [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_123, %.loc8_27.1 [concrete = constants.%int_123.5a3]
// CHECK:STDOUT:   %addr.loc8: %ptr.3a8 = addr_of %.loc8_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.1(%.loc8_27.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_30: init %C to %.loc8_3 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign %c1.var, %.loc8_30
// CHECK:STDOUT:   %.loc8_14: type = splice_block %C.ref.loc8_14 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_14: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %C = ref_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.479 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.479 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %C = var %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_3: ref %C = splice_block %c2.var {}
// CHECK:STDOUT:   %impl.elem0.loc9: %.698 = impl_witness_access constants.%Copy.impl_witness.bc7, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %true, %impl.elem0.loc9 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc9(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_34.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc9_34.2: ref bool = temporary %.loc9_34.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.e3b]
// CHECK:STDOUT:   %addr.loc9_38.1: %ptr.bb2 = addr_of %.loc9_34.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %addr.loc9_38.2: %ptr.3a8 = addr_of %.loc9_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.2(%addr.loc9_38.1, %addr.loc9_38.2)
// CHECK:STDOUT:   %.loc9_38: init %C to %.loc9_3 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   assign %c2.var, %.loc9_38
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %C = ref_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.479 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.479 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %C = var %c3.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc10_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %c1.ref: ref %C = name_ref c1, %c1
// CHECK:STDOUT:   %addr.loc10_34: %ptr.3a8 = addr_of %c1.ref
// CHECK:STDOUT:   %.loc10_3: ref %C = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc10_37: %ptr.3a8 = addr_of %.loc10_3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.3(%addr.loc10_34, %addr.loc10_37)
// CHECK:STDOUT:   %.loc10_37: init %C to %.loc10_3 = mark_in_place_init %C__carbon_thunk.call.loc10
// CHECK:STDOUT:   assign %c3.var, %.loc10_37
// CHECK:STDOUT:   %.loc10_21: type = splice_block %C.ref.loc10_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %C = ref_binding c3, %c3.var
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %c3.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%c3.var)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %c2.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%c2.var)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %c1.var, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%c1.var)
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
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bd2: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.980: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %As.type.c5c: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.fc5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3a8: %Core.IntLiteral.as.As.impl.Convert.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.33f: <witness> = impl_witness imports.%As.impl_witness_table.2e6, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2f4: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bc8: %Core.IntLiteral.as.As.impl.Convert.type.2f4 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.c5c = facet_value Core.IntLiteral, (%As.impl_witness.33f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.26c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.ca4: type = fn_type_with_self_type %As.WithSelf.Convert.type.26c, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.bc8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.bc8, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bde: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.6c4: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.fc5) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.3a8)]
// CHECK:STDOUT:   %As.impl_witness_table.2e6 = impl_witness_table (%Core.import_ref.6c4), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.479 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc8_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.bd2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %addr.loc8: %ptr.3a8 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc8_34.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_35.2: init %C to %.loc8_35.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_35.3: ref %C = temporary %.loc8_35.1, %.loc8_35.2
// CHECK:STDOUT:   %.loc8_35.4: %C = acquire_value %.loc8_35.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.479 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.ca4 = impl_witness_access constants.%As.impl_witness.33f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bc8]
// CHECK:STDOUT:   %bound_method.loc9_28.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_28.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.bde]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc9_28.2(%int_8.loc9) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_28.2: %i32 = converted %int_8.loc9, %.loc9_28.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_28.3: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.3a8 = addr_of %.loc9_28.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc9_28.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_28.4: init %C to %.loc9_28.3 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_28.5: init %C = converted %.loc9_28.2, %.loc9_28.4
// CHECK:STDOUT:   %.loc9_28.6: ref %C = temporary %.loc9_28.3, %.loc9_28.5
// CHECK:STDOUT:   %.loc9_28.7: %C = acquire_value %.loc9_28.6
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_28.7
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_28.6, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_28.6)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_35.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_implicit_multi_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.479: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.3a8: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.1: %C__carbon_thunk.type.0be6f8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bd2: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.980: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.299: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.e00: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.e6d: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.0be6f8.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.7c7210.2: %C__carbon_thunk.type.0be6f8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.c5c: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.fc5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3a8: %Core.IntLiteral.as.As.impl.Convert.type.fc5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.33f: <witness> = impl_witness imports.%As.impl_witness_table.2e6, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2f4: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bc8: %Core.IntLiteral.as.As.impl.Convert.type.2f4 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.c5c = facet_value Core.IntLiteral, (%As.impl_witness.33f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.26c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.ca4: type = fn_type_with_self_type %As.WithSelf.Convert.type.26c, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.bc8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.bc8, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bde: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.1: %C__carbon_thunk.type.0be6f8.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.7c7210.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.decl.5bf927.2: %C__carbon_thunk.type.0be6f8.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.7c7210.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.6c4: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.fc5) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.3a8)]
// CHECK:STDOUT:   %As.impl_witness_table.2e6 = impl_witness_table (%Core.import_ref.6c4), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.479 = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_38.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65]
// CHECK:STDOUT:   %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_34 [concrete = constants.%bound_method.bd2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%int_8.loc8) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_34 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_8.loc8, %.loc8_34.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %impl.elem0.loc8_37: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc8_37.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.299]
// CHECK:STDOUT:   %specific_fn.loc8_37: <specific function> = specific_function %impl.elem0.loc8_37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_37.2: <bound method> = bound_method %int_9, %specific_fn.loc8_37 [concrete = constants.%bound_method.e00]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37: init %i32 = call %bound_method.loc8_37.2(%int_9) [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %.loc8_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_37 [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %.loc8_37.2: %i32 = converted %int_9, %.loc8_37.1 [concrete = constants.%int_9.e6d]
// CHECK:STDOUT:   %addr.loc8: %ptr.3a8 = addr_of %.loc8_38.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.1(%.loc8_34.2, %.loc8_37.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_38.2: init %C to %.loc8_38.1 = mark_in_place_init %C__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_21: type = splice_block %C.ref.loc8_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.3: ref %C = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %C = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %c1: %C = value_binding c1, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.479 = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9_29: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc9_31: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_35.1: ref %C = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b65]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.bd2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_8.loc9) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_8.loc9, %.loc9_34.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %addr.loc9: %ptr.3a8 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %C__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.2(%.loc9_34.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_35.2: init %C to %.loc9_35.1 = mark_in_place_init %C__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_21: type = splice_block %C.ref.loc9_21 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_35.3: ref %C = temporary %.loc9_35.1, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: %C = acquire_value %.loc9_35.3
// CHECK:STDOUT:   %c2: %C = value_binding c2, %.loc9_35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.479 = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc10: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc10: %.ca4 = impl_witness_access constants.%As.impl_witness.33f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bc8]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %int_8.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %int_8.loc10, %specific_fn.loc10 [concrete = constants.%bound_method.bde]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc10_28.2(%int_8.loc10) [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc10_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc10_28.2: %i32 = converted %int_8.loc10, %.loc10_28.1 [concrete = constants.%int_8.980]
// CHECK:STDOUT:   %.loc10_21: type = splice_block %C.ref.loc10 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_28.3: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc10: %ptr.3a8 = addr_of %.loc10_28.3
// CHECK:STDOUT:   %C__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.5bf927.2(%.loc10_28.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_28.4: init %C to %.loc10_28.3 = mark_in_place_init %C__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_28.5: init %C = converted %.loc10_28.2, %.loc10_28.4
// CHECK:STDOUT:   %.loc10_28.6: ref %C = temporary %.loc10_28.3, %.loc10_28.5
// CHECK:STDOUT:   %.loc10_28.7: %C = acquire_value %.loc10_28.6
// CHECK:STDOUT:   %c3: %C = value_binding c3, %.loc10_28.7
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_28.6, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc10: init %empty_tuple.type = call %C.cpp_destructor.bound.loc10(%.loc10_28.6)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_35.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc9: init %empty_tuple.type = call %C.cpp_destructor.bound.loc9(%.loc9_35.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_38.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call.loc8: init %empty_tuple.type = call %C.cpp_destructor.bound.loc8(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

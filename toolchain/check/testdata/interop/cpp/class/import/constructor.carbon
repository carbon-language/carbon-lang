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

class DefaultC {
 public:
  DefaultC();
};

struct NoDefault {
  NoDefault(int);
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.DefaultC = Cpp.DefaultC.DefaultC();
  //@dump-sem-ir-end
}

// --- implicit_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  //@dump-sem-ir-begin
  var unused c: Cpp.DefaultC;
  //@dump-sem-ir-end
}

fn G(generic T: Core.Default & Core.Destroy) {
  var unused x: T;
}

fn CallG() {
  G(Cpp.DefaultC);
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

fn G(generic T: Core.Default & Core.Destroy) {
  var unused x: T;
}

fn CallG() {
  // CHECK:STDERR: fail_no_default.carbon:[[@LINE+3]]:3-18: error: cannot convert type `NoDefault` into type implementing `Core.Default & Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_no_default.carbon:[[@LINE-6]]:14-43: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  G(Cpp.NoDefault);
}

// ============================================================================
// Non default constructor
// ============================================================================

// --- non_default.h

class NonDefaultC {
 public:
  NonDefaultC(int x, int y);
};

// --- import_non_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_default.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c: Cpp.NonDefaultC = Cpp.NonDefaultC.NonDefaultC(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Multiple constructors
// ============================================================================

// --- multiple.h

class MultipleC {
 public:
  MultipleC();
  MultipleC(int x, int y);
};

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.MultipleC = Cpp.MultipleC.MultipleC();
  let unused c2: Cpp.MultipleC = Cpp.MultipleC.MultipleC(123, 456);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor with default arguments
// ============================================================================

// --- default_arguments.h

class DefaultArgsC {
 public:
  DefaultArgsC(int x, int y = 8);
};

// --- import_default_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "default_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.DefaultArgsC = Cpp.DefaultArgsC.DefaultArgsC(8, 9);
  let unused c2: Cpp.DefaultArgsC = Cpp.DefaultArgsC.DefaultArgsC(8);
  //@dump-sem-ir-end
}

// ============================================================================
// Constructor templates
// ============================================================================

// --- template.h

class TemplateC {
 public:
  TemplateC(int);
  template<typename T> TemplateC(T);
};

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "template.h";

fn F() {
  //@dump-sem-ir-begin
  var c1: Cpp.TemplateC = Cpp.TemplateC.TemplateC(123);
  var unused c2: Cpp.TemplateC = Cpp.TemplateC.TemplateC(true);
  var unused c3: Cpp.TemplateC = Cpp.TemplateC.TemplateC(&c1);
  //@dump-sem-ir-end
}

// ============================================================================
// Deleted constructor
// ============================================================================

// --- deleted.h

class DeletedC {
 public:
  DeletedC() = delete;
};

class DeletedD {
 public:
  DeletedD(int) = delete;
  DeletedD(short);
};

// --- fail_import_deleted.carbon

library "[[@TEST_NAME]]";

import Cpp library "deleted.h";

fn F() {
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+4]]:32-52: error: call to deleted function 'DeletedC' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_deleted.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./deleted.h:4:3-10: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:
  let unused c: Cpp.DeletedC = Cpp.DeletedC.DeletedC();
  // CHECK:STDERR: fail_import_deleted.carbon:[[@LINE+5]]:32-52: error: call to deleted function 'DeletedD' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_deleted.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./deleted.h:9:3-10: note: candidate constructor has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR: ./deleted.h:10:3-10: note: candidate constructor [CppInteropParseNote]
  // CHECK:STDERR:
  let unused d: Cpp.DeletedD = Cpp.DeletedD.DeletedD(0);
}

// ============================================================================
// Implicit single argument constructor
// ============================================================================

// --- implicit_single_argument.h

class SingleArgC {
 public:
  SingleArgC(int x);
};

// --- import_implicit_single_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_single_argument.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.SingleArgC = Cpp.SingleArgC.SingleArgC(8);
  let unused c2: Cpp.SingleArgC = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Implicit multi arguments constructor
// ============================================================================

// --- implicit_multi_arguments.h

class MultiArgC {
 public:
  MultiArgC(int x, int y = 8);
};

// --- import_implicit_multi_arguments.carbon

library "[[@TEST_NAME]]";

import Cpp library "implicit_multi_arguments.h";

fn F() {
  //@dump-sem-ir-begin
  let unused c1: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(8, 9);
  let unused c2: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(8);
  let unused c3: Cpp.MultiArgC = 8 as i32;
  //@dump-sem-ir-end
}

// ============================================================================
// Trying to call a constructor of an incomplete class
// ============================================================================

// --- incomplete.h

class IncompleteC;

// --- fail_import_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp library "incomplete.h";

fn F() {
   // CHECK:STDERR: fail_import_incomplete.carbon:[[@LINE+4]]:4-30: error: member access into incomplete class `Cpp.IncompleteC` [QualifiedExprInIncompleteClassScope]
   // CHECK:STDERR: included from: fail_import_incomplete.carbon:[[@LINE-4]]:10 [InCppInclude]
   // CHECK:STDERR: ./incomplete.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
   // CHECK:STDERR:
   Cpp.IncompleteC.IncompleteC();
}

// CHECK:STDOUT: --- import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultC: type = class_type @DefaultC [concrete]
// CHECK:STDOUT:   %pattern_type.903: type = pattern_type %DefaultC [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.903 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %DefaultC.DefaultC.cpp_overload_set.type: type = cpp_overload_set_type @DefaultC.DefaultC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefaultC.DefaultC.cpp_overload_set.value: %DefaultC.DefaultC.cpp_overload_set.type = cpp_overload_set_value @DefaultC.DefaultC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d69: type = ptr_type %DefaultC [concrete]
// CHECK:STDOUT:   %DefaultC__carbon_thunk.type: type = fn_type @DefaultC__carbon_thunk [concrete]
// CHECK:STDOUT:   %DefaultC__carbon_thunk: %DefaultC__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultC.cpp_destructor.type: type = fn_type @DefaultC.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefaultC.cpp_destructor: %DefaultC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefaultC = %DefaultC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultC.decl: type = class_decl @DefaultC [concrete = constants.%DefaultC] {} {}
// CHECK:STDOUT:   %DefaultC.DefaultC.cpp_overload_set.value: %DefaultC.DefaultC.cpp_overload_set.type = cpp_overload_set_value @DefaultC.DefaultC.cpp_overload_set [concrete = constants.%DefaultC.DefaultC.cpp_overload_set.value]
// CHECK:STDOUT:   %DefaultC__carbon_thunk.decl: %DefaultC__carbon_thunk.type = fn_decl @DefaultC__carbon_thunk [concrete = constants.%DefaultC__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DefaultC.ref.loc8_35: type = name_ref DefaultC, imports.%DefaultC.decl [concrete = constants.%DefaultC]
// CHECK:STDOUT:   %DefaultC.ref.loc8_44: %DefaultC.DefaultC.cpp_overload_set.type = name_ref DefaultC, imports.%DefaultC.DefaultC.cpp_overload_set.value [concrete = constants.%DefaultC.DefaultC.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_54.1: ref %DefaultC = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d69 = addr_of %.loc8_54.1
// CHECK:STDOUT:   %DefaultC__carbon_thunk.call: init %empty_tuple.type = call imports.%DefaultC__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_54.2: init %DefaultC to %.loc8_54.1 = mark_in_place_init %DefaultC__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_54.3: ref %DefaultC = temporary %.loc8_54.1, %.loc8_54.2
// CHECK:STDOUT:   %.loc8_54.4: %DefaultC = acquire_value %.loc8_54.3
// CHECK:STDOUT:   %.loc8_20: type = splice_block %DefaultC.ref.loc8_20 [concrete = constants.%DefaultC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultC.ref.loc8_20: type = name_ref DefaultC, imports.%DefaultC.decl [concrete = constants.%DefaultC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %DefaultC = wrapper_binding c, %.loc8_54.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.903 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultC.cpp_destructor.bound: <bound method> = bound_method %.loc8_54.3, constants.%DefaultC.cpp_destructor
// CHECK:STDOUT:   %DefaultC.cpp_destructor.call: init %empty_tuple.type = call %DefaultC.cpp_destructor.bound(%.loc8_54.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultC: type = class_type @DefaultC [concrete]
// CHECK:STDOUT:   %pattern_type.903: type = pattern_type %DefaultC [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.903 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.903 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultC.Op.type: type = fn_type @DefaultC.Op [concrete]
// CHECK:STDOUT:   %DefaultC.Op: %DefaultC.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%DefaultC.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.53e: %Default.type = facet_value %DefaultC, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.db1: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.53e) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.898: %DefaultOrUnformed.type = facet_value %DefaultC, (%DefaultOrUnformed.impl_witness.db1) [concrete]
// CHECK:STDOUT:   %DefaultC.cpp_destructor.type: type = fn_type @DefaultC.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefaultC.cpp_destructor: %DefaultC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefaultC = %DefaultC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultC.decl: type = class_decl @DefaultC [concrete = constants.%DefaultC] {} {}
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %DefaultC = var_storage %c.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%DefaultC, (constants.%DefaultOrUnformed.impl_witness.db1) [concrete = constants.%DefaultOrUnformed.facet.898]
// CHECK:STDOUT:   %.loc8_29.1: %DefaultOrUnformed.type = converted constants.%DefaultC, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet.898]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc8_29.1 [concrete = constants.%DefaultC]
// CHECK:STDOUT:   %.loc8_29.2: type = converted %.loc8_29.1, %as_type [concrete = constants.%DefaultC]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_3: ref %DefaultC = splice_block %c.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %DefaultC to %.loc8_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %c.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc8_20: type = splice_block %DefaultC.ref [concrete = constants.%DefaultC] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultC.ref: type = name_ref DefaultC, imports.%DefaultC.decl [concrete = constants.%DefaultC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %DefaultC = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.903 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.903 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultC.cpp_destructor.bound: <bound method> = bound_method %c.var, constants.%DefaultC.cpp_destructor
// CHECK:STDOUT:   %DefaultC.cpp_destructor.call: init %empty_tuple.type = call %DefaultC.cpp_destructor.bound(%c.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonDefaultC: type = class_type @NonDefaultC [concrete]
// CHECK:STDOUT:   %pattern_type.714: type = pattern_type %NonDefaultC [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.714 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %NonDefaultC.NonDefaultC.cpp_overload_set.type: type = cpp_overload_set_type @NonDefaultC.NonDefaultC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NonDefaultC.NonDefaultC.cpp_overload_set.value: %NonDefaultC.NonDefaultC.cpp_overload_set.type = cpp_overload_set_value @NonDefaultC.NonDefaultC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.9fb: type = ptr_type %NonDefaultC [concrete]
// CHECK:STDOUT:   %NonDefaultC__carbon_thunk.type: type = fn_type @NonDefaultC__carbon_thunk [concrete]
// CHECK:STDOUT:   %NonDefaultC__carbon_thunk: %NonDefaultC__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b1d: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.450: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.c5c: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.8dd: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %NonDefaultC.cpp_destructor.type: type = fn_type @NonDefaultC.cpp_destructor [concrete]
// CHECK:STDOUT:   %NonDefaultC.cpp_destructor: %NonDefaultC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonDefaultC = %NonDefaultC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonDefaultC.decl: type = class_decl @NonDefaultC [concrete = constants.%NonDefaultC] {} {}
// CHECK:STDOUT:   %NonDefaultC.NonDefaultC.cpp_overload_set.value: %NonDefaultC.NonDefaultC.cpp_overload_set.type = cpp_overload_set_value @NonDefaultC.NonDefaultC.cpp_overload_set [concrete = constants.%NonDefaultC.NonDefaultC.cpp_overload_set.value]
// CHECK:STDOUT:   %NonDefaultC__carbon_thunk.decl: %NonDefaultC__carbon_thunk.type = fn_decl @NonDefaultC__carbon_thunk [concrete = constants.%NonDefaultC__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NonDefaultC.ref.loc8_38: type = name_ref NonDefaultC, imports.%NonDefaultC.decl [concrete = constants.%NonDefaultC]
// CHECK:STDOUT:   %NonDefaultC.ref.loc8_50: %NonDefaultC.NonDefaultC.cpp_overload_set.type = name_ref NonDefaultC, imports.%NonDefaultC.NonDefaultC.cpp_overload_set.value [concrete = constants.%NonDefaultC.NonDefaultC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc8_71.1: ref %NonDefaultC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_63: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_63.1: <bound method> = bound_method %int_123, %impl.elem0.loc8_63 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b1d]
// CHECK:STDOUT:   %specific_fn.loc8_63: <specific function> = specific_function %impl.elem0.loc8_63, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_63.2: <bound method> = bound_method %int_123, %specific_fn.loc8_63 [concrete = constants.%bound_method.450]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_63: init %i32 = call %bound_method.loc8_63.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc8_63.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_63 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc8_63.2: %i32 = converted %int_123, %.loc8_63.1 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %impl.elem0.loc8_68: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_68.1: <bound method> = bound_method %int_456, %impl.elem0.loc8_68 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc]
// CHECK:STDOUT:   %specific_fn.loc8_68: <specific function> = specific_function %impl.elem0.loc8_68, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_68.2: <bound method> = bound_method %int_456, %specific_fn.loc8_68 [concrete = constants.%bound_method.c5c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_68: init %i32 = call %bound_method.loc8_68.2(%int_456) [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %.loc8_68.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_68 [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %.loc8_68.2: %i32 = converted %int_456, %.loc8_68.1 [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %addr: %ptr.9fb = addr_of %.loc8_71.1
// CHECK:STDOUT:   %NonDefaultC__carbon_thunk.call: init %empty_tuple.type = call imports.%NonDefaultC__carbon_thunk.decl(%.loc8_63.2, %.loc8_68.2, %addr)
// CHECK:STDOUT:   %.loc8_71.2: init %NonDefaultC to %.loc8_71.1 = mark_in_place_init %NonDefaultC__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_71.3: ref %NonDefaultC = temporary %.loc8_71.1, %.loc8_71.2
// CHECK:STDOUT:   %.loc8_71.4: %NonDefaultC = acquire_value %.loc8_71.3
// CHECK:STDOUT:   %.loc8_20: type = splice_block %NonDefaultC.ref.loc8_20 [concrete = constants.%NonDefaultC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %NonDefaultC.ref.loc8_20: type = name_ref NonDefaultC, imports.%NonDefaultC.decl [concrete = constants.%NonDefaultC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %NonDefaultC = wrapper_binding c, %.loc8_71.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.714 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonDefaultC.cpp_destructor.bound: <bound method> = bound_method %.loc8_71.3, constants.%NonDefaultC.cpp_destructor
// CHECK:STDOUT:   %NonDefaultC.cpp_destructor.call: init %empty_tuple.type = call %NonDefaultC.cpp_destructor.bound(%.loc8_71.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MultipleC: type = class_type @MultipleC [concrete]
// CHECK:STDOUT:   %pattern_type.56a: type = pattern_type %MultipleC [concrete]
// CHECK:STDOUT:   %c1.patt: %pattern_type.56a = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   %MultipleC.MultipleC.cpp_overload_set.type: type = cpp_overload_set_type @MultipleC.MultipleC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MultipleC.MultipleC.cpp_overload_set.value: %MultipleC.MultipleC.cpp_overload_set.type = cpp_overload_set_value @MultipleC.MultipleC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.898: type = ptr_type %MultipleC [concrete]
// CHECK:STDOUT:   %MultipleC__carbon_thunk.type.2f7b76.1: type = fn_type @MultipleC__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %MultipleC__carbon_thunk.dde1c4.1: %MultipleC__carbon_thunk.type.2f7b76.1 = struct_value () [concrete]
// CHECK:STDOUT:   %c2.patt: %pattern_type.56a = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %MultipleC__carbon_thunk.type.2f7b76.2: type = fn_type @MultipleC__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %MultipleC__carbon_thunk.dde1c4.2: %MultipleC__carbon_thunk.type.2f7b76.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b1d: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.450: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.c5c: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.8dd: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %MultipleC.cpp_destructor.type: type = fn_type @MultipleC.cpp_destructor [concrete]
// CHECK:STDOUT:   %MultipleC.cpp_destructor: %MultipleC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultipleC = %MultipleC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleC.decl: type = class_decl @MultipleC [concrete = constants.%MultipleC] {} {}
// CHECK:STDOUT:   %MultipleC.MultipleC.cpp_overload_set.value: %MultipleC.MultipleC.cpp_overload_set.type = cpp_overload_set_value @MultipleC.MultipleC.cpp_overload_set [concrete = constants.%MultipleC.MultipleC.cpp_overload_set.value]
// CHECK:STDOUT:   %MultipleC__carbon_thunk.decl.02182b.1: %MultipleC__carbon_thunk.type.2f7b76.1 = fn_decl @MultipleC__carbon_thunk.1 [concrete = constants.%MultipleC__carbon_thunk.dde1c4.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleC__carbon_thunk.decl.02182b.2: %MultipleC__carbon_thunk.type.2f7b76.2 = fn_decl @MultipleC__carbon_thunk.2 [concrete = constants.%MultipleC__carbon_thunk.dde1c4.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MultipleC.ref.loc8_37: type = name_ref MultipleC, imports.%MultipleC.decl [concrete = constants.%MultipleC]
// CHECK:STDOUT:   %MultipleC.ref.loc8_47: %MultipleC.MultipleC.cpp_overload_set.type = name_ref MultipleC, imports.%MultipleC.MultipleC.cpp_overload_set.value [concrete = constants.%MultipleC.MultipleC.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_58.1: ref %MultipleC = temporary_storage
// CHECK:STDOUT:   %addr.loc8: %ptr.898 = addr_of %.loc8_58.1
// CHECK:STDOUT:   %MultipleC__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%MultipleC__carbon_thunk.decl.02182b.1(%addr.loc8)
// CHECK:STDOUT:   %.loc8_58.2: init %MultipleC to %.loc8_58.1 = mark_in_place_init %MultipleC__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_58.3: ref %MultipleC = temporary %.loc8_58.1, %.loc8_58.2
// CHECK:STDOUT:   %.loc8_58.4: %MultipleC = acquire_value %.loc8_58.3
// CHECK:STDOUT:   %.loc8_21: type = splice_block %MultipleC.ref.loc8_21 [concrete = constants.%MultipleC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MultipleC.ref.loc8_21: type = name_ref MultipleC, imports.%MultipleC.decl [concrete = constants.%MultipleC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: %MultipleC = wrapper_binding c1, %.loc8_58.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.56a = value_binding_pattern c1 [concrete = constants.%c1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MultipleC.ref.loc9_37: type = name_ref MultipleC, imports.%MultipleC.decl [concrete = constants.%MultipleC]
// CHECK:STDOUT:   %MultipleC.ref.loc9_47: %MultipleC.MultipleC.cpp_overload_set.type = name_ref MultipleC, imports.%MultipleC.MultipleC.cpp_overload_set.value [concrete = constants.%MultipleC.MultipleC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:   %.loc9_66.1: ref %MultipleC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9_58: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_58.1: <bound method> = bound_method %int_123, %impl.elem0.loc9_58 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b1d]
// CHECK:STDOUT:   %specific_fn.loc9_58: <specific function> = specific_function %impl.elem0.loc9_58, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_58.2: <bound method> = bound_method %int_123, %specific_fn.loc9_58 [concrete = constants.%bound_method.450]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_58: init %i32 = call %bound_method.loc9_58.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc9_58.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_58 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc9_58.2: %i32 = converted %int_123, %.loc9_58.1 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %impl.elem0.loc9_63: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_63.1: <bound method> = bound_method %int_456, %impl.elem0.loc9_63 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc]
// CHECK:STDOUT:   %specific_fn.loc9_63: <specific function> = specific_function %impl.elem0.loc9_63, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_63.2: <bound method> = bound_method %int_456, %specific_fn.loc9_63 [concrete = constants.%bound_method.c5c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_63: init %i32 = call %bound_method.loc9_63.2(%int_456) [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %.loc9_63.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_63 [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %.loc9_63.2: %i32 = converted %int_456, %.loc9_63.1 [concrete = constants.%int_456.8dd]
// CHECK:STDOUT:   %addr.loc9: %ptr.898 = addr_of %.loc9_66.1
// CHECK:STDOUT:   %MultipleC__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%MultipleC__carbon_thunk.decl.02182b.2(%.loc9_58.2, %.loc9_63.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_66.2: init %MultipleC to %.loc9_66.1 = mark_in_place_init %MultipleC__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_66.3: ref %MultipleC = temporary %.loc9_66.1, %.loc9_66.2
// CHECK:STDOUT:   %.loc9_66.4: %MultipleC = acquire_value %.loc9_66.3
// CHECK:STDOUT:   %.loc9_21: type = splice_block %MultipleC.ref.loc9_21 [concrete = constants.%MultipleC] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MultipleC.ref.loc9_21: type = name_ref MultipleC, imports.%MultipleC.decl [concrete = constants.%MultipleC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: %MultipleC = wrapper_binding c2, %.loc9_66.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.56a = value_binding_pattern c2 [concrete = constants.%c2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultipleC.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_66.3, constants.%MultipleC.cpp_destructor
// CHECK:STDOUT:   %MultipleC.cpp_destructor.call.loc9: init %empty_tuple.type = call %MultipleC.cpp_destructor.bound.loc9(%.loc9_66.3)
// CHECK:STDOUT:   %MultipleC.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_58.3, constants.%MultipleC.cpp_destructor
// CHECK:STDOUT:   %MultipleC.cpp_destructor.call.loc8: init %empty_tuple.type = call %MultipleC.cpp_destructor.bound.loc8(%.loc8_58.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_default_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DefaultArgsC: type = class_type @DefaultArgsC [concrete]
// CHECK:STDOUT:   %pattern_type.52d: type = pattern_type %DefaultArgsC [concrete]
// CHECK:STDOUT:   %c1.patt: %pattern_type.52d = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   %DefaultArgsC.DefaultArgsC.cpp_overload_set.type: type = cpp_overload_set_type @DefaultArgsC.DefaultArgsC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DefaultArgsC.DefaultArgsC.cpp_overload_set.value: %DefaultArgsC.DefaultArgsC.cpp_overload_set.type = cpp_overload_set_value @DefaultArgsC.DefaultArgsC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.cf5: type = ptr_type %DefaultArgsC [concrete]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.type.4c24d7.1: type = fn_type @DefaultArgsC__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.137e61.1: %DefaultArgsC__carbon_thunk.type.4c24d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.01f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.9cf: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.9bf: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.056: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %c2.patt: %pattern_type.52d = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.type.4c24d7.2: type = fn_type @DefaultArgsC__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.137e61.2: %DefaultArgsC__carbon_thunk.type.4c24d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor.type: type = fn_type @DefaultArgsC.cpp_destructor [concrete]
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor: %DefaultArgsC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DefaultArgsC = %DefaultArgsC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultArgsC.decl: type = class_decl @DefaultArgsC [concrete = constants.%DefaultArgsC] {} {}
// CHECK:STDOUT:   %DefaultArgsC.DefaultArgsC.cpp_overload_set.value: %DefaultArgsC.DefaultArgsC.cpp_overload_set.type = cpp_overload_set_value @DefaultArgsC.DefaultArgsC.cpp_overload_set [concrete = constants.%DefaultArgsC.DefaultArgsC.cpp_overload_set.value]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.decl.45c66b.1: %DefaultArgsC__carbon_thunk.type.4c24d7.1 = fn_decl @DefaultArgsC__carbon_thunk.1 [concrete = constants.%DefaultArgsC__carbon_thunk.137e61.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.decl.45c66b.2: %DefaultArgsC__carbon_thunk.type.4c24d7.2 = fn_decl @DefaultArgsC__carbon_thunk.2 [concrete = constants.%DefaultArgsC__carbon_thunk.137e61.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DefaultArgsC.ref.loc8_40: type = name_ref DefaultArgsC, imports.%DefaultArgsC.decl [concrete = constants.%DefaultArgsC]
// CHECK:STDOUT:   %DefaultArgsC.ref.loc8_53: %DefaultArgsC.DefaultArgsC.cpp_overload_set.type = name_ref DefaultArgsC, imports.%DefaultArgsC.DefaultArgsC.cpp_overload_set.value [concrete = constants.%DefaultArgsC.DefaultArgsC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_71.1: ref %DefaultArgsC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_67: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_67.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_67 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd]
// CHECK:STDOUT:   %specific_fn.loc8_67: <specific function> = specific_function %impl.elem0.loc8_67, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_67.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_67 [concrete = constants.%bound_method.01f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_67: init %i32 = call %bound_method.loc8_67.2(%int_8.loc8) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_67.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_67 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_67.2: %i32 = converted %int_8.loc8, %.loc8_67.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %impl.elem0.loc8_70: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_70.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_70 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46]
// CHECK:STDOUT:   %specific_fn.loc8_70: <specific function> = specific_function %impl.elem0.loc8_70, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_70.2: <bound method> = bound_method %int_9, %specific_fn.loc8_70 [concrete = constants.%bound_method.9bf]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_70: init %i32 = call %bound_method.loc8_70.2(%int_9) [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc8_70.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_70 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc8_70.2: %i32 = converted %int_9, %.loc8_70.1 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %addr.loc8: %ptr.cf5 = addr_of %.loc8_71.1
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%DefaultArgsC__carbon_thunk.decl.45c66b.1(%.loc8_67.2, %.loc8_70.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_71.2: init %DefaultArgsC to %.loc8_71.1 = mark_in_place_init %DefaultArgsC__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_71.3: ref %DefaultArgsC = temporary %.loc8_71.1, %.loc8_71.2
// CHECK:STDOUT:   %.loc8_71.4: %DefaultArgsC = acquire_value %.loc8_71.3
// CHECK:STDOUT:   %.loc8_21: type = splice_block %DefaultArgsC.ref.loc8_21 [concrete = constants.%DefaultArgsC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultArgsC.ref.loc8_21: type = name_ref DefaultArgsC, imports.%DefaultArgsC.decl [concrete = constants.%DefaultArgsC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: %DefaultArgsC = wrapper_binding c1, %.loc8_71.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.52d = value_binding_pattern c1 [concrete = constants.%c1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DefaultArgsC.ref.loc9_40: type = name_ref DefaultArgsC, imports.%DefaultArgsC.decl [concrete = constants.%DefaultArgsC]
// CHECK:STDOUT:   %DefaultArgsC.ref.loc9_53: %DefaultArgsC.DefaultArgsC.cpp_overload_set.type = name_ref DefaultArgsC, imports.%DefaultArgsC.DefaultArgsC.cpp_overload_set.value [concrete = constants.%DefaultArgsC.DefaultArgsC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_68.1: ref %DefaultArgsC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_67.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_67.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.01f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_67.2(%int_8.loc9) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_67.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_67.2: %i32 = converted %int_8.loc9, %.loc9_67.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %addr.loc9: %ptr.cf5 = addr_of %.loc9_68.1
// CHECK:STDOUT:   %DefaultArgsC__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%DefaultArgsC__carbon_thunk.decl.45c66b.2(%.loc9_67.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_68.2: init %DefaultArgsC to %.loc9_68.1 = mark_in_place_init %DefaultArgsC__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_68.3: ref %DefaultArgsC = temporary %.loc9_68.1, %.loc9_68.2
// CHECK:STDOUT:   %.loc9_68.4: %DefaultArgsC = acquire_value %.loc9_68.3
// CHECK:STDOUT:   %.loc9_21: type = splice_block %DefaultArgsC.ref.loc9_21 [concrete = constants.%DefaultArgsC] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DefaultArgsC.ref.loc9_21: type = name_ref DefaultArgsC, imports.%DefaultArgsC.decl [concrete = constants.%DefaultArgsC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: %DefaultArgsC = wrapper_binding c2, %.loc9_68.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.52d = value_binding_pattern c2 [concrete = constants.%c2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_68.3, constants.%DefaultArgsC.cpp_destructor
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor.call.loc9: init %empty_tuple.type = call %DefaultArgsC.cpp_destructor.bound.loc9(%.loc9_68.3)
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_71.3, constants.%DefaultArgsC.cpp_destructor
// CHECK:STDOUT:   %DefaultArgsC.cpp_destructor.call.loc8: init %empty_tuple.type = call %DefaultArgsC.cpp_destructor.bound.loc8(%.loc8_71.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TemplateC: type = class_type @TemplateC [concrete]
// CHECK:STDOUT:   %pattern_type.f20: type = pattern_type %TemplateC [concrete]
// CHECK:STDOUT:   %c1.patt: %pattern_type.f20 = ref_binding_pattern c1 [concrete]
// CHECK:STDOUT:   %c1.var_patt: %pattern_type.f20 = var_pattern %c1.patt [concrete]
// CHECK:STDOUT:   %TemplateC.TemplateC.cpp_overload_set.type: type = cpp_overload_set_type @TemplateC.TemplateC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TemplateC.TemplateC.cpp_overload_set.value: %TemplateC.TemplateC.cpp_overload_set.type = cpp_overload_set_value @TemplateC.TemplateC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.588: type = ptr_type %TemplateC [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.type.7462cf.1: type = fn_type @TemplateC__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.a4007a.1: %TemplateC__carbon_thunk.type.7462cf.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %c2.patt: %pattern_type.f20 = ref_binding_pattern c2 [concrete]
// CHECK:STDOUT:   %c2.var_patt: %pattern_type.f20 = var_pattern %c2.patt [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.type.7462cf.2: type = fn_type @TemplateC__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.a4007a.2: %TemplateC__carbon_thunk.type.7462cf.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d02: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d02, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.e3b: ref bool = temporary invalid, %true [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.e3b [concrete]
// CHECK:STDOUT:   %c3.patt: %pattern_type.f20 = ref_binding_pattern c3 [concrete]
// CHECK:STDOUT:   %c3.var_patt: %pattern_type.f20 = var_pattern %c3.patt [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.type.7462cf.3: type = fn_type @TemplateC__carbon_thunk.3 [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.a4007a.3: %TemplateC__carbon_thunk.type.7462cf.3 = struct_value () [concrete]
// CHECK:STDOUT:   %TemplateC.cpp_destructor.type: type = fn_type @TemplateC.cpp_destructor [concrete]
// CHECK:STDOUT:   %TemplateC.cpp_destructor: %TemplateC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TemplateC = %TemplateC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TemplateC.decl: type = class_decl @TemplateC [concrete = constants.%TemplateC] {} {}
// CHECK:STDOUT:   %TemplateC.TemplateC.cpp_overload_set.value: %TemplateC.TemplateC.cpp_overload_set.type = cpp_overload_set_value @TemplateC.TemplateC.cpp_overload_set [concrete = constants.%TemplateC.TemplateC.cpp_overload_set.value]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.decl.a5784e.1: %TemplateC__carbon_thunk.type.7462cf.1 = fn_decl @TemplateC__carbon_thunk.1 [concrete = constants.%TemplateC__carbon_thunk.a4007a.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.decl.a5784e.2: %TemplateC__carbon_thunk.type.7462cf.2 = fn_decl @TemplateC__carbon_thunk.2 [concrete = constants.%TemplateC__carbon_thunk.a4007a.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %TemplateC__carbon_thunk.decl.a5784e.3: %TemplateC__carbon_thunk.type.7462cf.3 = fn_decl @TemplateC__carbon_thunk.3 [concrete = constants.%TemplateC__carbon_thunk.a4007a.3] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c1.var: ref %TemplateC = var_storage %c1.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TemplateC.ref.loc8_30: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   %TemplateC.ref.loc8_40: %TemplateC.TemplateC.cpp_overload_set.type = name_ref TemplateC, imports.%TemplateC.TemplateC.cpp_overload_set.value [concrete = constants.%TemplateC.TemplateC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %.loc8_3: ref %TemplateC = splice_block %c1.var {}
// CHECK:STDOUT:   %impl.elem0.loc8: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_51.1: <bound method> = bound_method %int_123, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_51.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_51.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc8_51.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc8_51.2: %i32 = converted %int_123, %.loc8_51.1 [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %addr.loc8: %ptr.588 = addr_of %.loc8_3
// CHECK:STDOUT:   %TemplateC__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%TemplateC__carbon_thunk.decl.a5784e.1(%.loc8_51.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_54: init %TemplateC to %.loc8_3 = mark_in_place_init %TemplateC__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign %c1.var, %.loc8_54
// CHECK:STDOUT:   %.loc8_14: type = splice_block %TemplateC.ref.loc8_14 [concrete = constants.%TemplateC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TemplateC.ref.loc8_14: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: ref %TemplateC = wrapper_binding c1, %c1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.f20 = ref_binding_pattern c1 [concrete = constants.%c1.patt]
// CHECK:STDOUT:     %c1.var_patt: %pattern_type.f20 = var_pattern %c1.patt [concrete = constants.%c1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2.var: ref %TemplateC = var_storage %c2.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc9_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TemplateC.ref.loc9_37: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   %TemplateC.ref.loc9_47: %TemplateC.TemplateC.cpp_overload_set.type = name_ref TemplateC, imports.%TemplateC.TemplateC.cpp_overload_set.value [concrete = constants.%TemplateC.TemplateC.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_3: ref %TemplateC = splice_block %c2.var {}
// CHECK:STDOUT:   %impl.elem0.loc9: %.e28 = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %true, %impl.elem0.loc9 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc9(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc9_58.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc9_58.2: ref bool = temporary %.loc9_58.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.e3b]
// CHECK:STDOUT:   %addr.loc9_62.1: %ptr.bb2 = addr_of %.loc9_58.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %addr.loc9_62.2: %ptr.588 = addr_of %.loc9_3
// CHECK:STDOUT:   %TemplateC__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%TemplateC__carbon_thunk.decl.a5784e.2(%addr.loc9_62.1, %addr.loc9_62.2)
// CHECK:STDOUT:   %.loc9_62: init %TemplateC to %.loc9_3 = mark_in_place_init %TemplateC__carbon_thunk.call.loc9
// CHECK:STDOUT:   assign %c2.var, %.loc9_62
// CHECK:STDOUT:   %.loc9_21: type = splice_block %TemplateC.ref.loc9_21 [concrete = constants.%TemplateC] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TemplateC.ref.loc9_21: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: ref %TemplateC = wrapper_binding c2, %c2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.f20 = ref_binding_pattern c2 [concrete = constants.%c2.patt]
// CHECK:STDOUT:     %c2.var_patt: %pattern_type.f20 = var_pattern %c2.patt [concrete = constants.%c2.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3.var: ref %TemplateC = var_storage %c3.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc10_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TemplateC.ref.loc10_37: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   %TemplateC.ref.loc10_47: %TemplateC.TemplateC.cpp_overload_set.type = name_ref TemplateC, imports.%TemplateC.TemplateC.cpp_overload_set.value [concrete = constants.%TemplateC.TemplateC.cpp_overload_set.value]
// CHECK:STDOUT:   %c1.ref: ref %TemplateC = name_ref c1, %c1
// CHECK:STDOUT:   %addr.loc10_58: %ptr.588 = addr_of %c1.ref
// CHECK:STDOUT:   %.loc10_3: ref %TemplateC = splice_block %c3.var {}
// CHECK:STDOUT:   %addr.loc10_61: %ptr.588 = addr_of %.loc10_3
// CHECK:STDOUT:   %TemplateC__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%TemplateC__carbon_thunk.decl.a5784e.3(%addr.loc10_58, %addr.loc10_61)
// CHECK:STDOUT:   %.loc10_61: init %TemplateC to %.loc10_3 = mark_in_place_init %TemplateC__carbon_thunk.call.loc10
// CHECK:STDOUT:   assign %c3.var, %.loc10_61
// CHECK:STDOUT:   %.loc10_21: type = splice_block %TemplateC.ref.loc10_21 [concrete = constants.%TemplateC] {
// CHECK:STDOUT:     %Cpp.ref.loc10_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %TemplateC.ref.loc10_21: type = name_ref TemplateC, imports.%TemplateC.decl [concrete = constants.%TemplateC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: ref %TemplateC = wrapper_binding c3, %c3.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.f20 = ref_binding_pattern c3 [concrete = constants.%c3.patt]
// CHECK:STDOUT:     %c3.var_patt: %pattern_type.f20 = var_pattern %c3.patt [concrete = constants.%c3.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TemplateC.cpp_destructor.bound.loc10: <bound method> = bound_method %c3.var, constants.%TemplateC.cpp_destructor
// CHECK:STDOUT:   %TemplateC.cpp_destructor.call.loc10: init %empty_tuple.type = call %TemplateC.cpp_destructor.bound.loc10(%c3.var)
// CHECK:STDOUT:   %TemplateC.cpp_destructor.bound.loc9: <bound method> = bound_method %c2.var, constants.%TemplateC.cpp_destructor
// CHECK:STDOUT:   %TemplateC.cpp_destructor.call.loc9: init %empty_tuple.type = call %TemplateC.cpp_destructor.bound.loc9(%c2.var)
// CHECK:STDOUT:   %TemplateC.cpp_destructor.bound.loc8: <bound method> = bound_method %c1.var, constants.%TemplateC.cpp_destructor
// CHECK:STDOUT:   %TemplateC.cpp_destructor.call.loc8: init %empty_tuple.type = call %TemplateC.cpp_destructor.bound.loc8(%c1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_implicit_single_argument.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %SingleArgC: type = class_type @SingleArgC [concrete]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %SingleArgC [concrete]
// CHECK:STDOUT:   %c1.patt: %pattern_type.bac = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   %SingleArgC.SingleArgC.cpp_overload_set.type: type = cpp_overload_set_type @SingleArgC.SingleArgC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %SingleArgC.SingleArgC.cpp_overload_set.value: %SingleArgC.SingleArgC.cpp_overload_set.type = cpp_overload_set_value @SingleArgC.SingleArgC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.15b: type = ptr_type %SingleArgC [concrete]
// CHECK:STDOUT:   %SingleArgC__carbon_thunk.type: type = fn_type @SingleArgC__carbon_thunk [concrete]
// CHECK:STDOUT:   %SingleArgC__carbon_thunk: %SingleArgC__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.01f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.9cf: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %c2.patt: %pattern_type.bac = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e7f: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.e7f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d02: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %As.WithSelf.Convert.type.d02, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.07f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %SingleArgC.cpp_destructor.type: type = fn_type @SingleArgC.cpp_destructor [concrete]
// CHECK:STDOUT:   %SingleArgC.cpp_destructor: %SingleArgC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .SingleArgC = %SingleArgC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SingleArgC.decl: type = class_decl @SingleArgC [concrete = constants.%SingleArgC] {} {}
// CHECK:STDOUT:   %SingleArgC.SingleArgC.cpp_overload_set.value: %SingleArgC.SingleArgC.cpp_overload_set.type = cpp_overload_set_value @SingleArgC.SingleArgC.cpp_overload_set [concrete = constants.%SingleArgC.SingleArgC.cpp_overload_set.value]
// CHECK:STDOUT:   %SingleArgC__carbon_thunk.decl: %SingleArgC__carbon_thunk.type = fn_decl @SingleArgC__carbon_thunk [concrete = constants.%SingleArgC__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %SingleArgC.ref.loc8_38: type = name_ref SingleArgC, imports.%SingleArgC.decl [concrete = constants.%SingleArgC]
// CHECK:STDOUT:   %SingleArgC.ref.loc8_49: %SingleArgC.SingleArgC.cpp_overload_set.type = name_ref SingleArgC, imports.%SingleArgC.SingleArgC.cpp_overload_set.value [concrete = constants.%SingleArgC.SingleArgC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc8_62.1: ref %SingleArgC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_61.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_61.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8 [concrete = constants.%bound_method.01f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_61.2(%int_8.loc8) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_61.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_61.2: %i32 = converted %int_8.loc8, %.loc8_61.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %addr.loc8: %ptr.15b = addr_of %.loc8_62.1
// CHECK:STDOUT:   %SingleArgC__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%SingleArgC__carbon_thunk.decl(%.loc8_61.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_62.2: init %SingleArgC to %.loc8_62.1 = mark_in_place_init %SingleArgC__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_62.3: ref %SingleArgC = temporary %.loc8_62.1, %.loc8_62.2
// CHECK:STDOUT:   %.loc8_62.4: %SingleArgC = acquire_value %.loc8_62.3
// CHECK:STDOUT:   %.loc8_21: type = splice_block %SingleArgC.ref.loc8_21 [concrete = constants.%SingleArgC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %SingleArgC.ref.loc8_21: type = name_ref SingleArgC, imports.%SingleArgC.decl [concrete = constants.%SingleArgC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: %SingleArgC = wrapper_binding c1, %.loc8_62.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.bac = value_binding_pattern c1 [concrete = constants.%c1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc9: %.453 = impl_witness_access constants.%As.impl_witness.e7f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc9_37.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_37.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.07f]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc9_37.2(%int_8.loc9) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_37.2: %i32 = converted %int_8.loc9, %.loc9_37.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_37.3: ref %SingleArgC = temporary_storage
// CHECK:STDOUT:   %addr.loc9: %ptr.15b = addr_of %.loc9_37.3
// CHECK:STDOUT:   %SingleArgC__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%SingleArgC__carbon_thunk.decl(%.loc9_37.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_37.4: init %SingleArgC to %.loc9_37.3 = mark_in_place_init %SingleArgC__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_37.5: init %SingleArgC = converted %.loc9_37.2, %.loc9_37.4
// CHECK:STDOUT:   %.loc9_37.6: ref %SingleArgC = temporary %.loc9_37.3, %.loc9_37.5
// CHECK:STDOUT:   %.loc9_37.7: %SingleArgC = acquire_value %.loc9_37.6
// CHECK:STDOUT:   %.loc9_21: type = splice_block %SingleArgC.ref.loc9 [concrete = constants.%SingleArgC] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %SingleArgC.ref.loc9: type = name_ref SingleArgC, imports.%SingleArgC.decl [concrete = constants.%SingleArgC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: %SingleArgC = wrapper_binding c2, %.loc9_37.7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.bac = value_binding_pattern c2 [concrete = constants.%c2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SingleArgC.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_37.6, constants.%SingleArgC.cpp_destructor
// CHECK:STDOUT:   %SingleArgC.cpp_destructor.call.loc9: init %empty_tuple.type = call %SingleArgC.cpp_destructor.bound.loc9(%.loc9_37.6)
// CHECK:STDOUT:   %SingleArgC.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_62.3, constants.%SingleArgC.cpp_destructor
// CHECK:STDOUT:   %SingleArgC.cpp_destructor.call.loc8: init %empty_tuple.type = call %SingleArgC.cpp_destructor.bound.loc8(%.loc8_62.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_implicit_multi_arguments.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MultiArgC: type = class_type @MultiArgC [concrete]
// CHECK:STDOUT:   %pattern_type.b5c: type = pattern_type %MultiArgC [concrete]
// CHECK:STDOUT:   %c1.patt: %pattern_type.b5c = value_binding_pattern c1 [concrete]
// CHECK:STDOUT:   %MultiArgC.MultiArgC.cpp_overload_set.type: type = cpp_overload_set_type @MultiArgC.MultiArgC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %MultiArgC.MultiArgC.cpp_overload_set.value: %MultiArgC.MultiArgC.cpp_overload_set.type = cpp_overload_set_value @MultiArgC.MultiArgC.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.b73: type = ptr_type %MultiArgC [concrete]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.type.e5b71e.1: type = fn_type @MultiArgC__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.dd06c5.1: %MultiArgC__carbon_thunk.type.e5b71e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.01f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.9cf: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.9bf: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.056: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %c2.patt: %pattern_type.b5c = value_binding_pattern c2 [concrete]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.type.e5b71e.2: type = fn_type @MultiArgC__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.dd06c5.2: %MultiArgC__carbon_thunk.type.e5b71e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %c3.patt: %pattern_type.b5c = value_binding_pattern c3 [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e7f: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.e7f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d02: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %As.WithSelf.Convert.type.d02, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.07f: <bound method> = bound_method %int_8.b85, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.type: type = fn_type @MultiArgC.cpp_destructor [concrete]
// CHECK:STDOUT:   %MultiArgC.cpp_destructor: %MultiArgC.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MultiArgC = %MultiArgC.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultiArgC.decl: type = class_decl @MultiArgC [concrete = constants.%MultiArgC] {} {}
// CHECK:STDOUT:   %MultiArgC.MultiArgC.cpp_overload_set.value: %MultiArgC.MultiArgC.cpp_overload_set.type = cpp_overload_set_value @MultiArgC.MultiArgC.cpp_overload_set [concrete = constants.%MultiArgC.MultiArgC.cpp_overload_set.value]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.decl.bde754.1: %MultiArgC__carbon_thunk.type.e5b71e.1 = fn_decl @MultiArgC__carbon_thunk.1 [concrete = constants.%MultiArgC__carbon_thunk.dd06c5.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.decl.bde754.2: %MultiArgC__carbon_thunk.type.e5b71e.2 = fn_decl @MultiArgC__carbon_thunk.2 [concrete = constants.%MultiArgC__carbon_thunk.dd06c5.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MultiArgC.ref.loc8_37: type = name_ref MultiArgC, imports.%MultiArgC.decl [concrete = constants.%MultiArgC]
// CHECK:STDOUT:   %MultiArgC.ref.loc8_47: %MultiArgC.MultiArgC.cpp_overload_set.type = name_ref MultiArgC, imports.%MultiArgC.MultiArgC.cpp_overload_set.value [concrete = constants.%MultiArgC.MultiArgC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc8_62.1: ref %MultiArgC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_58: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_58.1: <bound method> = bound_method %int_8.loc8, %impl.elem0.loc8_58 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd]
// CHECK:STDOUT:   %specific_fn.loc8_58: <specific function> = specific_function %impl.elem0.loc8_58, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_58.2: <bound method> = bound_method %int_8.loc8, %specific_fn.loc8_58 [concrete = constants.%bound_method.01f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_58: init %i32 = call %bound_method.loc8_58.2(%int_8.loc8) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_58.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_58 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc8_58.2: %i32 = converted %int_8.loc8, %.loc8_58.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %impl.elem0.loc8_61: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_61.1: <bound method> = bound_method %int_9, %impl.elem0.loc8_61 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f46]
// CHECK:STDOUT:   %specific_fn.loc8_61: <specific function> = specific_function %impl.elem0.loc8_61, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_61.2: <bound method> = bound_method %int_9, %specific_fn.loc8_61 [concrete = constants.%bound_method.9bf]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_61: init %i32 = call %bound_method.loc8_61.2(%int_9) [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc8_61.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_61 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %.loc8_61.2: %i32 = converted %int_9, %.loc8_61.1 [concrete = constants.%int_9.056]
// CHECK:STDOUT:   %addr.loc8: %ptr.b73 = addr_of %.loc8_62.1
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%MultiArgC__carbon_thunk.decl.bde754.1(%.loc8_58.2, %.loc8_61.2, %addr.loc8)
// CHECK:STDOUT:   %.loc8_62.2: init %MultiArgC to %.loc8_62.1 = mark_in_place_init %MultiArgC__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_62.3: ref %MultiArgC = temporary %.loc8_62.1, %.loc8_62.2
// CHECK:STDOUT:   %.loc8_62.4: %MultiArgC = acquire_value %.loc8_62.3
// CHECK:STDOUT:   %.loc8_21: type = splice_block %MultiArgC.ref.loc8_21 [concrete = constants.%MultiArgC] {
// CHECK:STDOUT:     %Cpp.ref.loc8_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MultiArgC.ref.loc8_21: type = name_ref MultiArgC, imports.%MultiArgC.decl [concrete = constants.%MultiArgC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c1: %MultiArgC = wrapper_binding c1, %.loc8_62.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c1.patt: %pattern_type.b5c = value_binding_pattern c1 [concrete = constants.%c1.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9_34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %MultiArgC.ref.loc9_37: type = name_ref MultiArgC, imports.%MultiArgC.decl [concrete = constants.%MultiArgC]
// CHECK:STDOUT:   %MultiArgC.ref.loc9_47: %MultiArgC.MultiArgC.cpp_overload_set.type = name_ref MultiArgC, imports.%MultiArgC.MultiArgC.cpp_overload_set.value [concrete = constants.%MultiArgC.MultiArgC.cpp_overload_set.value]
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc9_59.1: ref %MultiArgC = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_58.1: <bound method> = bound_method %int_8.loc9, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9dd]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_58.2: <bound method> = bound_method %int_8.loc9, %specific_fn.loc9 [concrete = constants.%bound_method.01f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_58.2(%int_8.loc9) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_58.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc9_58.2: %i32 = converted %int_8.loc9, %.loc9_58.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %addr.loc9: %ptr.b73 = addr_of %.loc9_59.1
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%MultiArgC__carbon_thunk.decl.bde754.2(%.loc9_58.2, %addr.loc9)
// CHECK:STDOUT:   %.loc9_59.2: init %MultiArgC to %.loc9_59.1 = mark_in_place_init %MultiArgC__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_59.3: ref %MultiArgC = temporary %.loc9_59.1, %.loc9_59.2
// CHECK:STDOUT:   %.loc9_59.4: %MultiArgC = acquire_value %.loc9_59.3
// CHECK:STDOUT:   %.loc9_21: type = splice_block %MultiArgC.ref.loc9_21 [concrete = constants.%MultiArgC] {
// CHECK:STDOUT:     %Cpp.ref.loc9_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MultiArgC.ref.loc9_21: type = name_ref MultiArgC, imports.%MultiArgC.decl [concrete = constants.%MultiArgC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c2: %MultiArgC = wrapper_binding c2, %.loc9_59.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c2.patt: %pattern_type.b5c = value_binding_pattern c2 [concrete = constants.%c2.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_8.loc10: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc10: %.453 = impl_witness_access constants.%As.impl_witness.e7f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc10_36.1: <bound method> = bound_method %int_8.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_36.2: <bound method> = bound_method %int_8.loc10, %specific_fn.loc10 [concrete = constants.%bound_method.07f]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc10_36.2(%int_8.loc10) [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc10_36.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc10_36.2: %i32 = converted %int_8.loc10, %.loc10_36.1 [concrete = constants.%int_8.9cf]
// CHECK:STDOUT:   %.loc10_36.3: ref %MultiArgC = temporary_storage
// CHECK:STDOUT:   %addr.loc10: %ptr.b73 = addr_of %.loc10_36.3
// CHECK:STDOUT:   %MultiArgC__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%MultiArgC__carbon_thunk.decl.bde754.2(%.loc10_36.2, %addr.loc10)
// CHECK:STDOUT:   %.loc10_36.4: init %MultiArgC to %.loc10_36.3 = mark_in_place_init %MultiArgC__carbon_thunk.call.loc10
// CHECK:STDOUT:   %.loc10_36.5: init %MultiArgC = converted %.loc10_36.2, %.loc10_36.4
// CHECK:STDOUT:   %.loc10_36.6: ref %MultiArgC = temporary %.loc10_36.3, %.loc10_36.5
// CHECK:STDOUT:   %.loc10_36.7: %MultiArgC = acquire_value %.loc10_36.6
// CHECK:STDOUT:   %.loc10_21: type = splice_block %MultiArgC.ref.loc10 [concrete = constants.%MultiArgC] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MultiArgC.ref.loc10: type = name_ref MultiArgC, imports.%MultiArgC.decl [concrete = constants.%MultiArgC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c3: %MultiArgC = wrapper_binding c3, %.loc10_36.7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c3.patt: %pattern_type.b5c = value_binding_pattern c3 [concrete = constants.%c3.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.bound.loc10: <bound method> = bound_method %.loc10_36.6, constants.%MultiArgC.cpp_destructor
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.call.loc10: init %empty_tuple.type = call %MultiArgC.cpp_destructor.bound.loc10(%.loc10_36.6)
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_59.3, constants.%MultiArgC.cpp_destructor
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.call.loc9: init %empty_tuple.type = call %MultiArgC.cpp_destructor.bound.loc9(%.loc9_59.3)
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_62.3, constants.%MultiArgC.cpp_destructor
// CHECK:STDOUT:   %MultiArgC.cpp_destructor.call.loc8: init %empty_tuple.type = call %MultiArgC.cpp_destructor.bound.loc8(%.loc8_62.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

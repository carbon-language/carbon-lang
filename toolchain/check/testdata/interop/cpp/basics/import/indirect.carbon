// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --share-cpp-ast
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/import/indirect.carbon

// --- shared.h

namespace A {
  struct X {
    X(int);
    int g();
    int x;
  };

  template<typename T>
  struct Y {
    T y;
  };

  struct Z {
    int z;
  };
}

// --- direct_import.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";

fn F() -> Cpp.A.X { return 1; }

fn G() -> Cpp.A.Y(i32);

fn H() -> Cpp.A.Y(Cpp.A.Z);

alias AX = Cpp.A.X;

// --- indirect_import_via_function.carbon

library "[[@TEST_NAME]]";

// TODO: Should be `export import Cpp ...`.
import Cpp library "shared.h";
import library "direct_import";

fn Field() -> i32 {
  //@dump-sem-ir-begin
  return F().x;
  //@dump-sem-ir-end
}

fn Function() -> i32 {
  //@dump-sem-ir-begin
  return F().g();
  //@dump-sem-ir-end
}

// --- not_included.carbon

library "[[@TEST_NAME]]";

import Cpp;
import library "direct_import";

fn Field() -> i32 {
  return F().x;
}

// --- fail_not_leaked.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
namespace A {
  struct X;
}
''';

fn TryToUseNotLeaked() {
  // TODO: Clang's diagnostic for using a class whose definition is not
  // reachable is confusing.
  // CHECK:STDERR: fail_not_leaked.carbon:[[@LINE+4]]:17-23: error: binding pattern has incomplete type `X` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: included from: direct_import.carbon:4:10 [InCppInclude]
  // CHECK:STDERR: ./shared.h:3:10: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  var unused x: Cpp.A.X = 0;
}

// --- no_cpp_import.carbon

library "[[@TEST_NAME]]";

import library "direct_import";

fn Field() -> i32 {
  return F().x;
}

// --- indirect_import_via_alias.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn Use() -> (i32, i32) {
  //@dump-sem-ir-begin
  var ax: AX = 1;
  return (ax.x, ax.g());
  //@dump-sem-ir-end
}

// --- import_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "shared.h";
import library "direct_import";

fn Use() -> i32 {
  //@dump-sem-ir-begin
  return G().y;
  //@dump-sem-ir-end
}

fn UseZ() -> Cpp.A.Z {
  //@dump-sem-ir-begin
  return H().y;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- indirect_import_via_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.85f: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3f1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.85f) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.3f1, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.type: type = cpp_overload_set_type @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.type: type = fn_type @X.g [concrete]
// CHECK:STDOUT:   %X.g: %X.g.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//direct_import, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %X.g.decl: %X.g.type = fn_decl @X.g [concrete = constants.%X.g] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Field() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_12.1: ref %X = temporary_storage
// CHECK:STDOUT:   %F.call: init %X to %.loc10_12.1 = call %F.ref()
// CHECK:STDOUT:   %.loc10_12.2: ref %X = temporary %.loc10_12.1, %F.call
// CHECK:STDOUT:   %x.ref: %X.elem = name_ref x, @X.%field_decl [concrete = @X.%field_decl]
// CHECK:STDOUT:   %.loc10_13.1: ref %i32 = class_element_access %.loc10_12.2, element0
// CHECK:STDOUT:   %.loc10_13.2: %i32 = acquire_value %.loc10_13.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc10_13.1: <bound method> = bound_method %.loc10_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_13.2: <bound method> = bound_method %.loc10_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_13.2(%.loc10_13.2)
// CHECK:STDOUT:   %Destroy.facet.loc10_12.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_12.3: %Destroy.type = converted constants.%X, %Destroy.facet.loc10_12.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_12.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_12.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc10_12.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Function() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc16_12.1: ref %X = temporary_storage
// CHECK:STDOUT:   %F.call: init %X to %.loc16_12.1 = call %F.ref()
// CHECK:STDOUT:   %.loc16_12.2: ref %X = temporary %.loc16_12.1, %F.call
// CHECK:STDOUT:   %g.ref: %X.g.cpp_overload_set.type = name_ref g, imports.%X.g.cpp_overload_set.value [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc16_12.2, %g.ref
// CHECK:STDOUT:   %.loc16_12.3: %X = acquire_value %.loc16_12.2
// CHECK:STDOUT:   %X.g.call: init %i32 = call imports.%X.g.decl(%.loc16_12.3)
// CHECK:STDOUT:   %Destroy.facet.loc16_12.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_12.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc16_12.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc16_12.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc16_12.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc16_12.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc16_12.2)
// CHECK:STDOUT:   return %X.g.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_import_via_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.88a: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %ax.patt: %pattern_type.88a = ref_binding_pattern ax [concrete]
// CHECK:STDOUT:   %ax.var_patt: %pattern_type.88a = var_pattern %ax.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.606: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %X__carbon_thunk.type: type = fn_type @X__carbon_thunk [concrete]
// CHECK:STDOUT:   %X__carbon_thunk: %X__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.type: type = cpp_overload_set_type @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.g.type: type = fn_type @X.g [concrete]
// CHECK:STDOUT:   %X.g: %X.g.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.85f: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3f1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.85f) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.3f1, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.AX: type = import_ref Main//direct_import, AX, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %X__carbon_thunk.decl: %X__carbon_thunk.type = fn_decl @X__carbon_thunk [concrete = constants.%X__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.g.cpp_overload_set.value: %X.g.cpp_overload_set.type = cpp_overload_set_value @X.g.cpp_overload_set [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %X.g.decl: %X.g.type = fn_decl @X.g [concrete = constants.%X.g] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() -> out %return.param: %tuple.type.35f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ax.var: ref %X = var_storage %ax.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc9_3.1: ref %X = splice_block %ax.var {}
// CHECK:STDOUT:   %impl.elem0.loc9: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_16.1: <bound method> = bound_method %int_1, %impl.elem0.loc9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_16.2: <bound method> = bound_method %int_1, %specific_fn.loc9 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_16.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %int_1, %.loc9_16.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %addr: %ptr.606 = addr_of %.loc9_3.1
// CHECK:STDOUT:   %X__carbon_thunk.call: init %empty_tuple.type = call imports.%X__carbon_thunk.decl(%.loc9_16.2, %addr)
// CHECK:STDOUT:   %.loc9_3.2: init %X to %.loc9_3.1 = mark_in_place_init %X__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_3.3: init %X = converted %int_1, %.loc9_3.2
// CHECK:STDOUT:   assign %ax.var, %.loc9_3.3
// CHECK:STDOUT:   %AX.ref: type = name_ref AX, imports.%Main.AX [concrete = constants.%X]
// CHECK:STDOUT:   %ax: ref %X = wrapper_binding ax, %ax.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ax.patt: %pattern_type.88a = ref_binding_pattern ax [concrete = constants.%ax.patt]
// CHECK:STDOUT:     %ax.var_patt: %pattern_type.88a = var_pattern %ax.patt [concrete = constants.%ax.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ax.ref.loc10_11: ref %X = name_ref ax, %ax
// CHECK:STDOUT:   %x.ref: %X.elem = name_ref x, @X.%field_decl [concrete = @X.%field_decl]
// CHECK:STDOUT:   %.loc10_13.1: ref %i32 = class_element_access %ax.ref.loc10_11, element0
// CHECK:STDOUT:   %ax.ref.loc10_17: ref %X = name_ref ax, %ax
// CHECK:STDOUT:   %g.ref: %X.g.cpp_overload_set.type = name_ref g, imports.%X.g.cpp_overload_set.value [concrete = constants.%X.g.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10_19: <bound method> = bound_method %ax.ref.loc10_17, %g.ref
// CHECK:STDOUT:   %X.g.call: init %i32 = call imports.%X.g.decl(%ax.ref.loc10_17)
// CHECK:STDOUT:   %.loc10_23.1: %tuple.type.35f = tuple_literal (%.loc10_13.1, %X.g.call)
// CHECK:STDOUT:   %.loc10_13.2: %i32 = acquire_value %.loc10_13.1
// CHECK:STDOUT:   %impl.elem0.loc10: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc10_13.1: <bound method> = bound_method %.loc10_13.2, %impl.elem0.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_13.2: <bound method> = bound_method %.loc10_13.2, %specific_fn.loc10
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_13.2(%.loc10_13.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc10_23.2: init %i32 to %tuple.elem0 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc10_23.3: init %i32 to %tuple.elem1 = in_place_init %X.g.call
// CHECK:STDOUT:   %.loc10_23.4: init %tuple.type.35f to %return.param = tuple_init (%.loc10_23.2, %.loc10_23.3)
// CHECK:STDOUT:   %.loc10_24: init %tuple.type.35f = converted %.loc10_23.1, %.loc10_23.4
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.3f1, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%ax.var)
// CHECK:STDOUT:   return %.loc10_24 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.90c: type = class_type @Y.1 [concrete]
// CHECK:STDOUT:   %Y.elem.9cb: type = unbound_element_type %Y.90c, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3fc: %SubobjectDestroy.type = facet_value %Y.90c, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.739: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3fc) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1f6: %Destroy.type = facet_value %Y.90c, (%DestroyImpl.impl_witness.739, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Y.e8d: type = class_type @Y.2 [concrete]
// CHECK:STDOUT:   %Y.elem.9a6: type = unbound_element_type %Y.e8d, %Z [concrete]
// CHECK:STDOUT:   %Z.Z.type: type = fn_type @Z.Z [concrete]
// CHECK:STDOUT:   %Z.Z: %Z.Z.type = struct_value () [concrete]
// CHECK:STDOUT:   %const.09e: type = const_type %Z [concrete]
// CHECK:STDOUT:   %ptr.5fc: type = ptr_type %const.09e [concrete]
// CHECK:STDOUT:   %ptr.df4: type = ptr_type %Z [concrete]
// CHECK:STDOUT:   %Z__carbon_thunk.type: type = fn_type @Z__carbon_thunk [concrete]
// CHECK:STDOUT:   %Z__carbon_thunk: %Z__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.Op.type: type = fn_type @Z.Op [concrete]
// CHECK:STDOUT:   %Z.Op: %Z.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f88: <witness> = custom_witness (%Z.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.ca1: %Copy.type = facet_value %Z, (%custom_witness.f88) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.983: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ca1) [concrete]
// CHECK:STDOUT:   %.3c7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.983, %Copy.facet.ca1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.dae: %SubobjectDestroy.type = facet_value %Y.e8d, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.980: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.dae) [concrete]
// CHECK:STDOUT:   %Destroy.facet.afc: %Destroy.type = facet_value %Y.e8d, (%DestroyImpl.impl_witness.980, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.G: %G.type = import_ref Main//direct_import, G, loaded [concrete = constants.%G]
// CHECK:STDOUT:   %Main.H: %H.type = import_ref Main//direct_import, H, loaded [concrete = constants.%H]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Z.Z.decl: %Z.Z.type = fn_decl @Z.Z [concrete = constants.%Z.Z] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z__carbon_thunk.decl: %Z__carbon_thunk.type = fn_decl @Z__carbon_thunk [concrete = constants.%Z__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%Main.G [concrete = constants.%G]
// CHECK:STDOUT:   %.loc9_12.1: ref %Y.90c = temporary_storage
// CHECK:STDOUT:   %G.call: init %Y.90c to %.loc9_12.1 = call %G.ref()
// CHECK:STDOUT:   %.loc9_12.2: ref %Y.90c = temporary %.loc9_12.1, %G.call
// CHECK:STDOUT:   %y.ref: %Y.elem.9cb = name_ref y, @Y.1.%field_decl [concrete = @Y.1.%field_decl]
// CHECK:STDOUT:   %.loc9_13.1: ref %i32 = class_element_access %.loc9_12.2, element0
// CHECK:STDOUT:   %.loc9_13.2: %i32 = acquire_value %.loc9_13.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc9_13.1: <bound method> = bound_method %.loc9_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_13.2: <bound method> = bound_method %.loc9_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc9_13.2(%.loc9_13.2)
// CHECK:STDOUT:   %Destroy.facet.loc9_12.1: %Destroy.type = facet_value constants.%Y.90c, (constants.%DestroyImpl.impl_witness.739, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.1f6]
// CHECK:STDOUT:   %.loc9_12.3: %Destroy.type = converted constants.%Y.90c, %Destroy.facet.loc9_12.1 [concrete = constants.%Destroy.facet.1f6]
// CHECK:STDOUT:   %Destroy.facet.loc9_12.2: %Destroy.type = facet_value constants.%Y.90c, (constants.%DestroyImpl.impl_witness.739, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.1f6]
// CHECK:STDOUT:   %.loc9_12.4: %Destroy.type = converted constants.%Y.90c, %Destroy.facet.loc9_12.2 [concrete = constants.%Destroy.facet.1f6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_12.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %Y.90c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseZ() -> out %return.param: %Z {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, imports.%Main.H [concrete = constants.%H]
// CHECK:STDOUT:   %.loc15_12.1: ref %Y.e8d = temporary_storage
// CHECK:STDOUT:   %H.call: init %Y.e8d to %.loc15_12.1 = call %H.ref()
// CHECK:STDOUT:   %.loc15_12.2: ref %Y.e8d = temporary %.loc15_12.1, %H.call
// CHECK:STDOUT:   %y.ref: %Y.elem.9a6 = name_ref y, @Y.2.%field_decl [concrete = @Y.2.%field_decl]
// CHECK:STDOUT:   %.loc15_13.1: ref %Z = class_element_access %.loc15_12.2, element0
// CHECK:STDOUT:   %.loc15_13.2: %Z = acquire_value %.loc15_13.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.3c7 = impl_witness_access constants.%custom_witness.f88, element0 [concrete = constants.%Z.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc15_13.2, %impl.elem0
// CHECK:STDOUT:   %.loc15_13.3: ref %Z = temporary_storage
// CHECK:STDOUT:   %Op.ref: %Z.Z.type = name_ref Op, imports.%Z.Z.decl [concrete = constants.%Z.Z]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_13.4: ref %Z = value_as_ref %.loc15_13.2
// CHECK:STDOUT:   %addr.loc15_13.1: %ptr.df4 = addr_of %.loc15_13.4
// CHECK:STDOUT:   %.loc15_13.5: %ptr.5fc = as_compatible %addr.loc15_13.1
// CHECK:STDOUT:   %.loc15_13.6: %ptr.5fc = converted %addr.loc15_13.1, %.loc15_13.5
// CHECK:STDOUT:   %addr.loc15_13.2: %ptr.df4 = addr_of %.loc13_19.1
// CHECK:STDOUT:   %Z__carbon_thunk.call: init %empty_tuple.type = call imports.%Z__carbon_thunk.decl(%.loc15_13.6, %addr.loc15_13.2)
// CHECK:STDOUT:   %.loc15_13.7: init %Z to %.loc13_19.1 = mark_in_place_init %Z__carbon_thunk.call
// CHECK:STDOUT:   %Destroy.facet.loc15_12.1: %Destroy.type = facet_value constants.%Y.e8d, (constants.%DestroyImpl.impl_witness.980, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.afc]
// CHECK:STDOUT:   %.loc15_12.3: %Destroy.type = converted constants.%Y.e8d, %Destroy.facet.loc15_12.1 [concrete = constants.%Destroy.facet.afc]
// CHECK:STDOUT:   %Destroy.facet.loc15_12.2: %Destroy.type = facet_value constants.%Y.e8d, (constants.%DestroyImpl.impl_witness.980, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.afc]
// CHECK:STDOUT:   %.loc15_12.4: %Destroy.type = converted constants.%Y.e8d, %Destroy.facet.loc15_12.2 [concrete = constants.%Destroy.facet.afc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc15_12.2)
// CHECK:STDOUT:   return %.loc15_13.7 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15(%self.param: ref %Y.e8d) = "no_op";
// CHECK:STDOUT:

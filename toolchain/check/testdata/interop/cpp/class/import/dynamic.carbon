// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/dynamic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/dynamic.carbon

// --- dynamic_base_from_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''

struct FurtherBase {
  virtual auto further_base_func() -> int {
    return 11;
  }
};

struct Base: FurtherBase {
  int i;
  virtual auto func() -> int {
    return 7;
  }
};

auto Use(Base& b) -> int {
  return b.func();
}

''';

base class Derived {
  extend base: Cpp.Base;
  virtual fn other_func(unused self) -> i32 {
    return 3;
  }
}

//@dump-sem-ir-begin
inline Cpp '''

// TODO: Carbon lowering should request clang to generate the vtable.
void WorkaroundToEmitCtorAndVtable() {
  Carbon::Derived d;
}

// TODO: Carbon lowering should emit initialization that refers to the
// Clang-lowered vtable.
auto DoThing() -> int {
  Carbon::Derived d;
  return Use(d);
}

''';
//@dump-sem-ir-end

// CHECK:STDOUT: --- dynamic_base_from_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %FurtherBase.further_base_func.type: type = fn_type @FurtherBase.further_base_func [concrete]
// CHECK:STDOUT:   %FurtherBase.further_base_func: %FurtherBase.further_base_func.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.func.type: type = fn_type @Base.func [concrete]
// CHECK:STDOUT:   %Base.func: %Base.func.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %FurtherBase.further_base_func.decl: %FurtherBase.further_base_func.type = fn_decl @FurtherBase.further_base_func [concrete = constants.%FurtherBase.further_base_func] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.func.decl: %Base.func.type = fn_decl @Base.func [concrete = constants.%Base.func] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.ref.loc35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   inline_cpp "\n// TODO: Carbon lowering should request clang to generate the vtable.\nvoid WorkaroundToEmitCtorAndVtable() {\n  Carbon::Derived d;\n}\n\n// TODO: Carbon lowering should emit initialization that refers to the\n// Clang-lowered vtable.\nauto DoThing() -> int {\n  Carbon::Derived d;\n  return Use(d);\n}\n\n"
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   imports.%FurtherBase.further_base_func.decl
// CHECK:STDOUT:   imports.%Base.func.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @FurtherBase.vtable {
// CHECK:STDOUT:   imports.%FurtherBase.further_base_func.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%FurtherBase.further_base_func
// CHECK:STDOUT:   constants.%Base.func
// CHECK:STDOUT:   @Derived.%Derived.other_func.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:

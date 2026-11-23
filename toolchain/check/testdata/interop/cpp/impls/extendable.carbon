// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/extendable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/extendable.carbon

// --- class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};
''';

fn Test() {
  //@dump-sem-ir-begin
  Cpp.Class as Core.Extendable;
  //@dump-sem-ir-end
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Abstract {
 public:
  virtual void F() = 0;
};
''';

fn Test() {
  Cpp.Abstract as Core.Extendable;
}

// --- dynamic_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Dynamic {
 public:
  virtual void F();
};
''';

fn Test() {
  Cpp.Dynamic as Core.Extendable;
}

// --- derived_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};
class Derived : public Class {};
''';

fn Test() {
  Cpp.Derived as Core.Extendable;
}

// --- class_template.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};

template <typename T>
class Template {};
''';

fn Test() {
  Cpp.Template(Cpp.Class) as Core.Extendable;
}

// --- fail_final_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Final final {};
''';

fn Test() {
  Cpp.Final as Core.Extendable;
}

// --- fail_abstract_final_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
// CHECK:STDERR:        ╭─┤ fail_abstract_final_class.carbon:[[@LINE+11]]:7
// CHECK:STDERR: ->     │ class AbstractFinal final {
// CHECK:STDERR:        ·       ┳
// CHECK:STDERR:        ·       ┗━━┫ warning: abstract class is marked 'final' [CppInteropParseWarning]
// CHECK:STDERR:        │  public:
// CHECK:STDERR:        │   virtual void F() = 0;
// CHECK:STDERR:        ·                ┬
// CHECK:STDERR:        ·                ╰──┤ unimplemented pure virtual method 'F' in 'AbstractFinal'
// CHECK:STDERR:        ·                   │ [CppInteropParseNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class AbstractFinal final {
 public:
  virtual void F() = 0;
};
''';

fn Test() {
  Cpp.AbstractFinal as Core.Extendable;
}

// --- fail_virtual_base.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};
class VirtualBase : public virtual Class {};
''';

fn Test() {
  Cpp.VirtualBase as Core.Extendable;
}

// --- fail_union.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
union Union {};
''';

fn Test() {
  Cpp.Union as Core.Extendable;
}

// --- fail_enum.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
enum Enum { A };
''';

fn Test() {
  Cpp.Enum as Core.Extendable;
}

// --- fail_scoped_enum.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
enum class ScopedEnum { A };
''';

fn Test() {
  Cpp.ScopedEnum as Core.Extendable;
}

// --- fail_incomplete_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Incomplete;
''';

fn Test() {
  Cpp.Incomplete as Core.Extendable;
}

// --- generic.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};
''';

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: Cpp.Class) -> partial Cpp.Class* {
  return F(ref c);
}

// --- fail_generic_final_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Final final {};
''';

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: Cpp.Final) {
  // CHECK:STDERR:        ╭─┤ fail_generic_final_class.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        │ fn F[T: Core.Extendable](ref t: T) -> partial T* {
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref c);
  // CHECK:STDERR:        ·   ┳━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Cpp.Final` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref c);
}

// --- fail_impl_final_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Final final {};
''';

// CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+8]]:1
// CHECK:STDERR: ->     │ impl Cpp.Final as Core.Extendable {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: explicit impl of `Core.Extendable`; implemented automatically for
// CHECK:STDERR:        ·    ┃        `base` and `abstract` classes
// CHECK:STDERR:        ·    ┃        [ImplOfExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl Cpp.Final as Core.Extendable {}

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: Cpp.Final) {
  // CHECK:STDERR:        ╭─┤ fail_impl_final_class.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        │ fn F[T: Core.Extendable](ref t: T) -> partial T* {
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref c);
  // CHECK:STDERR:        ·   ┳━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Cpp.Final` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref c);
}

// CHECK:STDOUT: --- class.carbon = $class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %Extendable.type: type = facet_type <@Extendable> [concrete]
// CHECK:STDOUT:   %custom_witness: <witness> = custom_witness (), @Extendable [concrete]
// CHECK:STDOUT:   %Extendable.facet: %Extendable.type = facet_value %Class, (%custom_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Extendable = %Core.Extendable
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Core.Extendable: type = import_ref Core//prelude/parts/destroy, Extendable, loaded [concrete = constants.%Extendable.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, imports.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %Extendable.ref: type = name_ref Extendable, imports.%Core.Extendable [concrete = constants.%Extendable.type]
// CHECK:STDOUT:   %Extendable.facet: %Extendable.type = facet_value %Class.ref, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc9: %Extendable.type = converted %Class.ref, %Extendable.facet [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

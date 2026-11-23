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
  Cpp.Class as Core.Extendable;
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
  // CHECK:STDERR:        ╭─┤ fail_final_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.Final as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Final` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
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
  // CHECK:STDERR:        ╭─┤ fail_abstract_final_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.AbstractFinal as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `AbstractFinal` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.AbstractFinal as Core.Extendable;
}

// --- fail_virtual_base.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Class {};
class VirtualBase : public virtual Class {};
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_virtual_base.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.VirtualBase as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `VirtualBase` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.VirtualBase as Core.Extendable;
}

// --- fail_union.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
union Union {};
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_union.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.Union as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Union` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.Union as Core.Extendable;
}

// --- fail_enum.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
enum Enum { A };
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_enum.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.Enum as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Enum` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.Enum as Core.Extendable;
}

// --- fail_scoped_enum.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
enum class ScopedEnum { A };
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_scoped_enum.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.ScopedEnum as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `ScopedEnum` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.ScopedEnum as Core.Extendable;
}

// --- fail_incomplete_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Incomplete;
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_incomplete_class.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   Cpp.Incomplete as Core.Extendable;
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `Incomplete` into type implementing
  // CHECK:STDERR:        ·      ┃        `Core.Extendable`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
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

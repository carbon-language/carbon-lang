// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE

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
  F(ref c);
}

// --- fail_impl_final_class.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
class Final final {};
''';

impl Cpp.Final as Core.Extendable {}

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn G(ref c: Cpp.Final) {
  F(ref c);
}

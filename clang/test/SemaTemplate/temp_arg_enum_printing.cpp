// RUN: %clang_cc1 -fsyntax-only -ast-print %s | FileCheck %s

namespace NamedEnumNS
{
  
enum NamedEnum
{
  Val0,
  Val1
};
  
template <NamedEnum E>
void foo();
  
void test() {
  // CHECK: template <NamedEnumNS::NamedEnum E = NamedEnumNS::NamedEnum::Val0>
  NamedEnumNS::foo<Val0>();
  // CHECK: template <NamedEnumNS::NamedEnum E = NamedEnumNS::NamedEnum::Val1>
  NamedEnumNS::foo<(NamedEnum)1>();
  // CHECK: template <NamedEnumNS::NamedEnum E = 2>
  NamedEnumNS::foo<(NamedEnum)2>();
}
  
} // NamedEnumNS

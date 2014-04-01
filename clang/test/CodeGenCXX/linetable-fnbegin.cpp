// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// Test that the line table info for Foo<T>::bar() is pointing to the
// right header file.
// CHECK: define{{.*}}bar
// CHECK-NOT: define
// CHECK: ret {{.*}}, !dbg [[DBG:.*]]
// CHECK: [[HPP:.*]] = metadata !{metadata !"./template.hpp",
// CHECK: [[SP:.*]] = metadata !{i32 786478, metadata [[HPP]], metadata !"_ZTS3FooIiE", metadata !"bar", metadata !"bar", metadata !"_ZN3FooIiE3barEv", i32 22, metadata !8, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%class.Foo*)* @_ZN3FooIiE3barEv, null, metadata !7, metadata !2, i32 22} ; [ DW_TAG_subprogram ] [line 22] [def] [bar]
// We shouldn't need a lexical block for this function.
// CHECK: [[DBG]] = metadata !{i32 23, i32 0, metadata [[SP]], null}


# 1 "./template.h" 1
template <typename T>
class Foo {
public:
  int bar();
};
# 21 "./template.hpp"
template <typename T>
int Foo<T>::bar() {
  return 23;
}
int main (int argc, const char * argv[])
{
  Foo<int> f;
  return f.bar();
}

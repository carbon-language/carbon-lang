// RUN: %clang_cc1 -triple %itanium_abi_triple %s -emit-llvm -o - | FileCheck %s

inline void InlineFunc() {}
// CHECK: define linkonce_odr void @_Z10InlineFuncv() #[[INLINEHINTATTR:[0-9]+]] comdat {

struct MyClass {
  static void InlineStaticMethod();
  void InlineInstanceMethod();
};
inline void MyClass::InlineStaticMethod() {}
// CHECK: define linkonce_odr void @_ZN7MyClass18InlineStaticMethodEv() #[[INLINEHINTATTR]] comdat
inline void MyClass::InlineInstanceMethod() {}
// CHECK: define linkonce_odr void @_ZN7MyClass20InlineInstanceMethodEv(%struct.MyClass* %this) #[[INLINEHINTATTR]] comdat

template <typename T>
struct MyTemplate {
  static void InlineStaticMethod();
  void InlineInstanceMethod();
};
template <typename T> inline void MyTemplate<T>::InlineStaticMethod() {}
// CHECK: define linkonce_odr void @_ZN10MyTemplateIiE18InlineStaticMethodEv() #[[INLINEHINTATTR]] comdat
template <typename T> inline void MyTemplate<T>::InlineInstanceMethod() {}
// CHECK: define linkonce_odr void @_ZN10MyTemplateIiE20InlineInstanceMethodEv(%struct.MyTemplate* %this) #[[INLINEHINTATTR]] comdat

void UseThem() {
  InlineFunc();
  MyClass::InlineStaticMethod();
  MyClass().InlineInstanceMethod();
  MyTemplate<int>::InlineStaticMethod();
  MyTemplate<int>().InlineInstanceMethod();
}

// CHECK: attributes #[[INLINEHINTATTR]] = { {{.*}}inlinehint{{.*}} }

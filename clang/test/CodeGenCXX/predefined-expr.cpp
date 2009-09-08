// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

// CHECK: @__func__.externFunction = private constant [15 x i8] c"externFunction\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::externFunction()" = private constant [26 x i8] c"void NS::externFunction()\00"

// CHECK: @__func__.classTemplateFunction = private constant [22 x i8] c"classTemplateFunction\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::ClassTemplate<NS::Base *>::classTemplateFunction()" = private constant [60 x i8] c"void NS::ClassTemplate<NS::Base *>::classTemplateFunction()\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::ClassTemplate<int>::classTemplateFunction()" = private constant [53 x i8] c"void NS::ClassTemplate<int>::classTemplateFunction()\00"

// CHECK: @__func__.functionTemplate1 = private constant [18 x i8] c"functionTemplate1\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::functionTemplate1(NS::Base *)" = private constant [45 x i8] c"void NS::Base::functionTemplate1(NS::Base *)\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::functionTemplate1(int)" = private constant [38 x i8] c"void NS::Base::functionTemplate1(int)\00"

// CHECK: @"__func__.~Destructor" = private constant [12 x i8] c"~Destructor\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Destructor::~Destructor()" = private constant [35 x i8] c"void NS::Destructor::~Destructor()\00"

// CHECK: @__func__.Constructor = private constant [12 x i8] c"Constructor\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Constructor::Constructor(NS::Base *)" = private constant [46 x i8] c"void NS::Constructor::Constructor(NS::Base *)\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Constructor::Constructor(int)" = private constant [39 x i8] c"void NS::Constructor::Constructor(int)\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Constructor::Constructor()" = private constant [36 x i8] c"void NS::Constructor::Constructor()\00"

// CHECK: @__func__.virtualFunction = private constant [16 x i8] c"virtualFunction\00"
// CHECK: @"__PRETTY_FUNCTION__.virtual void NS::Derived::virtualFunction()" = private constant [44 x i8] c"virtual void NS::Derived::virtualFunction()\00"

// CHECK: @__func__.functionReturingTemplate2 = private constant [26 x i8] c"functionReturingTemplate2\00"
// CHECK: @"__PRETTY_FUNCTION__.ClassTemplate<NS::Base *> NS::Base::functionReturingTemplate2()" = private constant [64 x i8] c"ClassTemplate<NS::Base *> NS::Base::functionReturingTemplate2()\00"

// CHECK: @__func__.functionReturingTemplate1 = private constant [26 x i8] c"functionReturingTemplate1\00"
// CHECK: @"__PRETTY_FUNCTION__.ClassTemplate<int> NS::Base::functionReturingTemplate1()" = private constant [57 x i8] c"ClassTemplate<int> NS::Base::functionReturingTemplate1()\00"

// CHECK: @__func__.withTemplateParameter2 = private constant [23 x i8] c"withTemplateParameter2\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::withTemplateParameter2(ClassTemplate<NS::Base *>)" = private constant [65 x i8] c"void NS::Base::withTemplateParameter2(ClassTemplate<NS::Base *>)\00"

// CHECK: @__func__.withTemplateParameter1 = private constant [23 x i8] c"withTemplateParameter1\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::withTemplateParameter1(ClassTemplate<int>)" = private constant [58 x i8] c"void NS::Base::withTemplateParameter1(ClassTemplate<int>)\00"

// CHECK: @__func__.functionReturningClass = private constant [23 x i8] c"functionReturningClass\00"
// CHECK: @"__PRETTY_FUNCTION__.NS::Base *NS::Base::functionReturningClass()" = private constant [45 x i8] c"NS::Base *NS::Base::functionReturningClass()\00"

// CHECK: @__func__.functionWithParameters = private constant [23 x i8] c"functionWithParameters\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::functionWithParameters(int, float *, NS::Base *)" = private constant [64 x i8] c"void NS::Base::functionWithParameters(int, float *, NS::Base *)\00"

// CHECK: @__func__.variadicFunction = private constant [17 x i8] c"variadicFunction\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::variadicFunction(int, ...)" = private constant [42 x i8] c"void NS::Base::variadicFunction(int, ...)\00"

// CHECK: @"__PRETTY_FUNCTION__.virtual void NS::Base::virtualFunction()" = private constant [41 x i8] c"virtual void NS::Base::virtualFunction()\00"

// CHECK: @__func__.inlineFunction = private constant [15 x i8] c"inlineFunction\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::inlineFunction()" = private constant [32 x i8] c"void NS::Base::inlineFunction()\00"

// CHECK: @__func__.staticFunc = private constant [11 x i8] c"staticFunc\00"
// CHECK: @"__PRETTY_FUNCTION__.void NS::Base::staticFunc()" = private constant [28 x i8] c"void NS::Base::staticFunc()\00"

#include <stdio.h>

namespace NS {

template<typename T>
class ClassTemplate {
public:
  void classTemplateFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Base {
public:
  static void staticFunc() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  inline void inlineFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  virtual void virtualFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void functionWithParameters(int, float*, Base* base) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Base *functionReturningClass() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return 0;
  }

  void variadicFunction(int, ...) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void withTemplateParameter1(ClassTemplate<int>) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void withTemplateParameter2(ClassTemplate<Base *>) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  ClassTemplate<int> functionReturingTemplate1() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return ClassTemplate<int>();
  }

  ClassTemplate<Base *> functionReturingTemplate2() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return ClassTemplate<Base *>();
  }

  template<typename T>
  void functionTemplate1(T t) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Derived : public Base {
public:
  // Virtual function without being explicitally written.
  void virtualFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Constructor {
public:
  Constructor() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Constructor(int) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Constructor(Base *) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

};

class Destructor {
public:
  ~Destructor() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

extern void externFunction() {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

}

int main() {
  NS::Base::staticFunc();

  NS::Base b;
  b.inlineFunction();
  b.virtualFunction();
  b.variadicFunction(0);
  b.functionWithParameters(0, 0, 0);
  b.functionReturningClass();
  
  b.withTemplateParameter1(NS::ClassTemplate<int>());
  b.withTemplateParameter2(NS::ClassTemplate<NS::Base *>());
  b.functionReturingTemplate1();
  b.functionReturingTemplate2();
  b.functionTemplate1<int>(0);
  b.functionTemplate1<NS::Base *>(0);
  
  NS::Derived d;
  d.virtualFunction();
  
  NS::ClassTemplate<int> t1;
  t1.classTemplateFunction();
  NS::ClassTemplate<NS::Base *> t2;
  t2.classTemplateFunction();
  
  NS::Constructor c1;
  NS::Constructor c2(0);
  NS::Constructor c3((NS::Base *)0);
  
  {
    NS::Destructor destructor;
  }
  
  NS::externFunction();
  
  return 0;
}

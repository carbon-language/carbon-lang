// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: private unnamed_addr constant [15 x i8] c"externFunction\00"
// CHECK: private unnamed_addr constant [26 x i8] c"void NS::externFunction()\00"

// CHECK: private unnamed_addr constant [22 x i8] c"classTemplateFunction\00"
// CHECK: private unnamed_addr constant [60 x i8] c"void NS::ClassTemplate<NS::Base *>::classTemplateFunction()\00"
// CHECK: private unnamed_addr constant [53 x i8] c"void NS::ClassTemplate<int>::classTemplateFunction()\00"

// CHECK: private unnamed_addr constant [18 x i8] c"functionTemplate1\00"
// CHECK: private unnamed_addr constant [45 x i8] c"void NS::Base::functionTemplate1(NS::Base *)\00"
// CHECK: private unnamed_addr constant [38 x i8] c"void NS::Base::functionTemplate1(int)\00"

// CHECK: private unnamed_addr constant [23 x i8] c"anonymousUnionFunction\00"
// CHECK: private unnamed_addr constant [83 x i8] c"void NS::ContainerForAnonymousRecords::<anonymous union>::anonymousUnionFunction()\00"

// CHECK: private unnamed_addr constant [24 x i8] c"anonymousStructFunction\00"
// CHECK: private unnamed_addr constant [85 x i8] c"void NS::ContainerForAnonymousRecords::<anonymous struct>::anonymousStructFunction()\00"

// CHECK: private unnamed_addr constant [23 x i8] c"anonymousClassFunction\00"
// CHECK: private unnamed_addr constant [83 x i8] c"void NS::ContainerForAnonymousRecords::<anonymous class>::anonymousClassFunction()\00"

// CHECK: private unnamed_addr constant [12 x i8] c"~Destructor\00"
// CHECK: private unnamed_addr constant [30 x i8] c"NS::Destructor::~Destructor()\00"

// CHECK: private unnamed_addr constant [12 x i8] c"Constructor\00"
// CHECK: private unnamed_addr constant [41 x i8] c"NS::Constructor::Constructor(NS::Base *)\00"
// CHECK: private unnamed_addr constant [34 x i8] c"NS::Constructor::Constructor(int)\00"
// CHECK: private unnamed_addr constant [31 x i8] c"NS::Constructor::Constructor()\00"

// CHECK: private unnamed_addr constant [16 x i8] c"virtualFunction\00"
// CHECK: private unnamed_addr constant [44 x i8] c"virtual void NS::Derived::virtualFunction()\00"

// CHECK: private unnamed_addr constant [22 x i8] c"constVolatileFunction\00"
// CHECK: private unnamed_addr constant [54 x i8] c"void NS::Base::constVolatileFunction() const volatile\00"

// CHECK: private unnamed_addr constant [17 x i8] c"volatileFunction\00"
// CHECK: private unnamed_addr constant [43 x i8] c"void NS::Base::volatileFunction() volatile\00"

// CHECK: private unnamed_addr constant [14 x i8] c"constFunction\00"
// CHECK: private unnamed_addr constant [37 x i8] c"void NS::Base::constFunction() const\00"

// CHECK: private unnamed_addr constant [26 x i8] c"functionReturingTemplate2\00"
// CHECK: private unnamed_addr constant [64 x i8] c"ClassTemplate<NS::Base *> NS::Base::functionReturingTemplate2()\00"

// CHECK: private unnamed_addr constant [26 x i8] c"functionReturingTemplate1\00"
// CHECK: private unnamed_addr constant [57 x i8] c"ClassTemplate<int> NS::Base::functionReturingTemplate1()\00"

// CHECK: private unnamed_addr constant [23 x i8] c"withTemplateParameter2\00"
// CHECK: private unnamed_addr constant [65 x i8] c"void NS::Base::withTemplateParameter2(ClassTemplate<NS::Base *>)\00"

// CHECK: private unnamed_addr constant [23 x i8] c"withTemplateParameter1\00"
// CHECK: private unnamed_addr constant [58 x i8] c"void NS::Base::withTemplateParameter1(ClassTemplate<int>)\00"

// CHECK: private unnamed_addr constant [23 x i8] c"functionReturningClass\00"
// CHECK: private unnamed_addr constant [45 x i8] c"NS::Base *NS::Base::functionReturningClass()\00"

// CHECK: private unnamed_addr constant [23 x i8] c"functionWithParameters\00"
// CHECK: private unnamed_addr constant [64 x i8] c"void NS::Base::functionWithParameters(int, float *, NS::Base *)\00"

// CHECK: private unnamed_addr constant [17 x i8] c"variadicFunction\00"
// CHECK: private unnamed_addr constant [42 x i8] c"void NS::Base::variadicFunction(int, ...)\00"

// CHECK: private unnamed_addr constant [41 x i8] c"virtual void NS::Base::virtualFunction()\00"

// CHECK: private unnamed_addr constant [15 x i8] c"inlineFunction\00"
// CHECK: private unnamed_addr constant [32 x i8] c"void NS::Base::inlineFunction()\00"

// CHECK: private unnamed_addr constant [15 x i8] c"staticFunction\00"
// CHECK: private unnamed_addr constant [39 x i8] c"static void NS::Base::staticFunction()\00"

// CHECK: private unnamed_addr constant [26 x i8] c"topLevelNamespaceFunction\00"
// CHECK: private unnamed_addr constant [59 x i8] c"void ClassInTopLevelNamespace::topLevelNamespaceFunction()\00"

// CHECK: private unnamed_addr constant [27 x i8] c"anonymousNamespaceFunction\00"
// CHECK: private unnamed_addr constant [84 x i8] c"void <anonymous namespace>::ClassInAnonymousNamespace::anonymousNamespaceFunction()\00"

// CHECK: private unnamed_addr constant [19 x i8] c"localClassFunction\00"
// CHECK: private unnamed_addr constant [59 x i8] c"void NS::localClass(int)::LocalClass::localClassFunction()\00"

int printf(const char * _Format, ...);

class ClassInTopLevelNamespace {
public:
  void topLevelNamespaceFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

namespace {

  class ClassInAnonymousNamespace {
  public:
    void anonymousNamespaceFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  };

} // end anonymous namespace

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
  static void staticFunction() {
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

  void constFunction() const {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void volatileFunction() volatile {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
  
  void constVolatileFunction() const volatile {
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

class ContainerForAnonymousRecords {
public:
  class {
  public:
    void anonymousClassFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousClass;

  struct {
    void anonymousStructFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousStruct;

  union {
    void anonymousUnionFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousUnion;
};

void localClass(int) {
  class LocalClass {
  public:
    void localClassFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  };
  LocalClass lc;
  lc.localClassFunction();
}

extern void externFunction() {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

} // end NS namespace

int main() {
  ClassInAnonymousNamespace anonymousNamespace;
  anonymousNamespace.anonymousNamespaceFunction();

  ClassInTopLevelNamespace topLevelNamespace;
  topLevelNamespace.topLevelNamespaceFunction();

  NS::Base::staticFunction();
  
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
  b.constFunction();
  b.volatileFunction();
  b.constVolatileFunction();

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

  NS::ContainerForAnonymousRecords anonymous; 
  anonymous.anonymousClass.anonymousClassFunction();
  anonymous.anonymousStruct.anonymousStructFunction();
  anonymous.anonymousUnion.anonymousUnionFunction();

  NS::localClass(0);

  NS::externFunction();

  return 0;
}

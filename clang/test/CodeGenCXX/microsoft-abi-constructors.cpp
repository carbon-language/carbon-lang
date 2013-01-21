// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

class A {
 public:
  A() { }
  ~A() { }
};

void no_contstructor_destructor_infinite_recursion() {
  A a;

// CHECK:      define linkonce_odr x86_thiscallcc %class.A* @"\01??0A@@QAE@XZ"(%class.A* %this)
// CHECK:        [[THIS_ADDR:%[.0-9A-Z_a-z]+]] = alloca %class.A*, align 4
// CHECK-NEXT:   store %class.A* %this, %class.A** [[THIS_ADDR]], align 4
// CHECK-NEXT:   [[T1:%[.0-9A-Z_a-z]+]] = load %class.A** [[THIS_ADDR]]
// CHECK-NEXT:   ret %class.A* [[T1]]
// CHECK-NEXT: }

// Make sure that the destructor doesn't call itself:
// CHECK: define {{.*}} @"\01??1A@@QAE@XZ"
// CHECK-NOT: call void @"\01??1A@@QAE@XZ"
// CHECK: ret
}

struct B {
  virtual ~B();
  virtual void foo();
};

void check_vftable_offset() {
  B b;
// The vftable pointer should point at the beginning of the vftable.
// CHECK: [[THIS_PTR:%[0-9]+]] = bitcast %struct.B* {{.*}} to i8***
// CHECK: store i8** getelementptr inbounds ([2 x i8*]* @"\01??_7B@@6B@", i64 0, i64 0), i8*** [[THIS_PTR]]
}

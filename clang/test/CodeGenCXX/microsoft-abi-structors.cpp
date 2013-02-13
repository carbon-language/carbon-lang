// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 -fno-rtti > %t 2>&1
// RUN: FileCheck %s < %t
// Using a different check prefix as the inline destructors might be placed
// anywhere in the output.
// RUN: FileCheck --check-prefix=DTORS %s < %t

class A {
 public:
  A() { }
  ~A() { }
};

void no_constructor_destructor_infinite_recursion() {
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
  virtual ~B() {
// Complete destructor first:
// DTORS: define {{.*}} x86_thiscallcc void @"\01??1B@@UAE@XZ"(%struct.B* %this)

// Then, the scalar deleting destructor (used in the vtable):
// DTORS:      define {{.*}} x86_thiscallcc void @"\01??_GB@@UAEPAXI@Z"(%struct.B* %this, i1 zeroext %should_call_delete)
// DTORS:        %[[FROMBOOL:[0-9a-z]+]] = zext i1 %should_call_delete to i8
// DTORS-NEXT:   store i8 %[[FROMBOOL]], i8* %[[SHOULD_DELETE_VAR:[0-9a-z]+]], align 1
// DTORS:        %[[SHOULD_DELETE_VALUE:[0-9a-z]+]] = load i8* %[[SHOULD_DELETE_VAR]]
// DTORS:        call x86_thiscallcc void @"\01??1B@@UAE@XZ"(%struct.B* %[[THIS:[0-9a-z]+]])
// DTORS-NEXT:   %[[CONDITION:[0-9]+]] = icmp eq i8 %[[SHOULD_DELETE_VALUE]], 0
// DTORS-NEXT:   br i1 %[[CONDITION]], label %[[CONTINUE_LABEL:[0-9a-z._]+]], label %[[CALL_DELETE_LABEL:[0-9a-z._]+]]
//
// DTORS:      [[CALL_DELETE_LABEL]]
// DTORS-NEXT:   %[[THIS_AS_VOID:[0-9a-z]+]] = bitcast %struct.B* %[[THIS]] to i8*
// DTORS-NEXT:   call void @"\01??3@YAXPAX@Z"(i8* %[[THIS_AS_VOID]]) nounwind
// DTORS-NEXT:   br label %[[CONTINUE_LABEL]]
//
// DTORS:      [[CONTINUE_LABEL]]
// DTORS-NEXT:   ret void
  }
  virtual void foo();
};

// Emits the vftable in the output.
void B::foo() {}

void check_vftable_offset() {
  B b;
// The vftable pointer should point at the beginning of the vftable.
// CHECK: [[THIS_PTR:%[0-9]+]] = bitcast %struct.B* {{.*}} to i8***
// CHECK: store i8** getelementptr inbounds ([2 x i8*]* @"\01??_7B@@6B@", i64 0, i64 0), i8*** [[THIS_PTR]]
}

// FIXME: Enable the following block and add expectations when calls
// to virtual complete dtor are supported.
#if 0
void call_complete_dtor(B *obj_ptr) {
  obj_ptr->~B();
}
#endif

void call_deleting_dtor(B *obj_ptr) {
// FIXME: Add CHECKs when calls to virtual deleting dtor are generated properly.
  delete obj_ptr;
}

struct C {
  static int foo();

  C() {
    static int ctor_static = foo();
    // CHECK that the static in the ctor gets mangled correctly:
    // CHECK: @"\01?ctor_static@?1???0C@@QAE@XZ@4HA"
  }
  ~C() {
    static int dtor_static = foo();
    // CHECK that the static in the dtor gets mangled correctly:
    // CHECK: @"\01?dtor_static@?1???1C@@QAE@XZ@4HA"
  }
};

void use_C() { C c; }

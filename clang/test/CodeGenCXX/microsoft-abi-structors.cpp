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
// FIXME: add a test that verifies that the out-of-line scalar deleting
// destructor is linkonce_odr too.
// DTORS:      define linkonce_odr x86_thiscallcc void @"\01??_GB@@UAEPAXI@Z"(%struct.B* %this, i1 zeroext %should_call_delete)
// DTORS:        %[[FROMBOOL:[0-9a-z]+]] = zext i1 %should_call_delete to i8
// DTORS-NEXT:   store i8 %[[FROMBOOL]], i8* %[[SHOULD_DELETE_VAR:[0-9a-z._]+]], align 1
// DTORS:        %[[SHOULD_DELETE_VALUE:[0-9a-z._]+]] = load i8* %[[SHOULD_DELETE_VAR]]
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

void call_complete_dtor(B *obj_ptr) {
// CHECK: define void @"\01?call_complete_dtor@@YAXPAUB@@@Z"(%struct.B* %obj_ptr)
  obj_ptr->~B();
// CHECK: %[[OBJ_PTR_VALUE:.*]] = load %struct.B** %{{.*}}, align 4
// CHECK-NEXT: %[[PVTABLE:.*]] = bitcast %struct.B* %[[OBJ_PTR_VALUE]] to void (%struct.B*, i1)***
// CHECK-NEXT: %[[VTABLE:.*]] = load void (%struct.B*, i1)*** %[[PVTABLE]]
// CHECK-NEXT: %[[PVDTOR:.*]] = getelementptr inbounds void (%struct.B*, i1)** %[[VTABLE]], i64 0
// CHECK-NEXT: %[[VDTOR:.*]] = load void (%struct.B*, i1)** %[[PVDTOR]]
// CHECK-NEXT: call x86_thiscallcc void %[[VDTOR]](%struct.B* %[[OBJ_PTR_VALUE]], i1 zeroext false)
// CHECK-NEXT: ret void
}

void call_deleting_dtor(B *obj_ptr) {
// CHECK: define void @"\01?call_deleting_dtor@@YAXPAUB@@@Z"(%struct.B* %obj_ptr)
  delete obj_ptr;
// CHECK:      %[[OBJ_PTR_VALUE:.*]] = load %struct.B** %{{.*}}, align 4
// CHECK:      {{.*}}:{{%{0,1}[0-9]*}}
// CHECK-NEXT:   %[[PVTABLE:.*]] = bitcast %struct.B* %[[OBJ_PTR_VALUE]] to void (%struct.B*, i1)***
// CHECK-NEXT:   %[[VTABLE:.*]] = load void (%struct.B*, i1)*** %[[PVTABLE]]
// CHECK-NEXT:   %[[PVDTOR:.*]] = getelementptr inbounds void (%struct.B*, i1)** %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[VDTOR:.*]] = load void (%struct.B*, i1)** %[[PVDTOR]]
// CHECK-NEXT:   call x86_thiscallcc void %[[VDTOR]](%struct.B* %[[OBJ_PTR_VALUE]], i1 zeroext true)
// CHECK:      ret void
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

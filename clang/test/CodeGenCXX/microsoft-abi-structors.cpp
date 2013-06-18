// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 -fno-rtti > %t 2>&1
// RUN: FileCheck %s < %t
// Using a different check prefix as the inline destructors might be placed
// anywhere in the output.
// RUN: FileCheck --check-prefix=DTORS %s < %t

namespace basic {

class A {
 public:
  A() { }
  ~A() { }
};

void no_constructor_destructor_infinite_recursion() {
  A a;

// CHECK:      define linkonce_odr x86_thiscallcc %"class.basic::A"* @"\01??0A@basic@@QAE@XZ"(%"class.basic::A"* returned %this)
// CHECK:        [[THIS_ADDR:%[.0-9A-Z_a-z]+]] = alloca %"class.basic::A"*, align 4
// CHECK-NEXT:   store %"class.basic::A"* %this, %"class.basic::A"** [[THIS_ADDR]], align 4
// CHECK-NEXT:   [[T1:%[.0-9A-Z_a-z]+]] = load %"class.basic::A"** [[THIS_ADDR]]
// CHECK-NEXT:   ret %"class.basic::A"* [[T1]]
// CHECK-NEXT: }

// Make sure that the destructor doesn't call itself:
// CHECK: define {{.*}} @"\01??1A@basic@@QAE@XZ"
// CHECK-NOT: call void @"\01??1A@basic@@QAE@XZ"
// CHECK: ret
}

struct B {
  B();
};

// Tests that we can define constructors outside the class (PR12784).
B::B() {
  // CHECK: define x86_thiscallcc %"struct.basic::B"* @"\01??0B@basic@@QAE@XZ"(%"struct.basic::B"* returned %this)
  // CHECK: ret
}

struct C {
  virtual ~C() {
// Complete destructor first:
// DTORS: define {{.*}} x86_thiscallcc void @"\01??1C@basic@@UAE@XZ"(%"struct.basic::C"* %this)

// Then, the scalar deleting destructor (used in the vtable):
// FIXME: add a test that verifies that the out-of-line scalar deleting
// destructor is linkonce_odr too.
// DTORS:      define linkonce_odr x86_thiscallcc void @"\01??_GC@basic@@UAEPAXI@Z"(%"struct.basic::C"* %this, i1 zeroext %should_call_delete)
// DTORS:        %[[FROMBOOL:[0-9a-z]+]] = zext i1 %should_call_delete to i8
// DTORS-NEXT:   store i8 %[[FROMBOOL]], i8* %[[SHOULD_DELETE_VAR:[0-9a-z._]+]], align 1
// DTORS:        %[[SHOULD_DELETE_VALUE:[0-9a-z._]+]] = load i8* %[[SHOULD_DELETE_VAR]]
// DTORS:        call x86_thiscallcc void @"\01??1C@basic@@UAE@XZ"(%"struct.basic::C"* %[[THIS:[0-9a-z]+]])
// DTORS-NEXT:   %[[CONDITION:[0-9]+]] = icmp eq i8 %[[SHOULD_DELETE_VALUE]], 0
// DTORS-NEXT:   br i1 %[[CONDITION]], label %[[CONTINUE_LABEL:[0-9a-z._]+]], label %[[CALL_DELETE_LABEL:[0-9a-z._]+]]
//
// DTORS:      [[CALL_DELETE_LABEL]]
// DTORS-NEXT:   %[[THIS_AS_VOID:[0-9a-z]+]] = bitcast %"struct.basic::C"* %[[THIS]] to i8*
// DTORS-NEXT:   call void @"\01??3@YAXPAX@Z"(i8* %[[THIS_AS_VOID]]) [[NUW:#[0-9]+]]
// DTORS-NEXT:   br label %[[CONTINUE_LABEL]]
//
// DTORS:      [[CONTINUE_LABEL]]
// DTORS-NEXT:   ret void
  }
  virtual void foo();
};

// Emits the vftable in the output.
void C::foo() {}

void check_vftable_offset() {
  C c;
// The vftable pointer should point at the beginning of the vftable.
// CHECK: [[THIS_PTR:%[0-9]+]] = bitcast %"struct.basic::C"* {{.*}} to i8***
// CHECK: store i8** getelementptr inbounds ([2 x i8*]* @"\01??_7C@basic@@6B@", i64 0, i64 0), i8*** [[THIS_PTR]]
}

void call_complete_dtor(C *obj_ptr) {
// CHECK: define void @"\01?call_complete_dtor@basic@@YAXPAUC@1@@Z"(%"struct.basic::C"* %obj_ptr)
  obj_ptr->~C();
// CHECK: %[[OBJ_PTR_VALUE:.*]] = load %"struct.basic::C"** %{{.*}}, align 4
// CHECK-NEXT: %[[PVTABLE:.*]] = bitcast %"struct.basic::C"* %[[OBJ_PTR_VALUE]] to void (%"struct.basic::C"*, i1)***
// CHECK-NEXT: %[[VTABLE:.*]] = load void (%"struct.basic::C"*, i1)*** %[[PVTABLE]]
// CHECK-NEXT: %[[PVDTOR:.*]] = getelementptr inbounds void (%"struct.basic::C"*, i1)** %[[VTABLE]], i64 0
// CHECK-NEXT: %[[VDTOR:.*]] = load void (%"struct.basic::C"*, i1)** %[[PVDTOR]]
// CHECK-NEXT: call x86_thiscallcc void %[[VDTOR]](%"struct.basic::C"* %[[OBJ_PTR_VALUE]], i1 zeroext false)
// CHECK-NEXT: ret void
}

void call_deleting_dtor(C *obj_ptr) {
// CHECK: define void @"\01?call_deleting_dtor@basic@@YAXPAUC@1@@Z"(%"struct.basic::C"* %obj_ptr)
  delete obj_ptr;
// CHECK:      %[[OBJ_PTR_VALUE:.*]] = load %"struct.basic::C"** %{{.*}}, align 4
// CHECK:      br i1 {{.*}}, label %[[DELETE_NULL:.*]], label %[[DELETE_NOTNULL:.*]]

// CHECK:      [[DELETE_NOTNULL]]
// CHECK-NEXT:   %[[PVTABLE:.*]] = bitcast %"struct.basic::C"* %[[OBJ_PTR_VALUE]] to void (%"struct.basic::C"*, i1)***
// CHECK-NEXT:   %[[VTABLE:.*]] = load void (%"struct.basic::C"*, i1)*** %[[PVTABLE]]
// CHECK-NEXT:   %[[PVDTOR:.*]] = getelementptr inbounds void (%"struct.basic::C"*, i1)** %[[VTABLE]], i64 0
// CHECK-NEXT:   %[[VDTOR:.*]] = load void (%"struct.basic::C"*, i1)** %[[PVDTOR]]
// CHECK-NEXT:   call x86_thiscallcc void %[[VDTOR]](%"struct.basic::C"* %[[OBJ_PTR_VALUE]], i1 zeroext true)
// CHECK:      ret void
}

struct D {
  static int foo();

  D() {
    static int ctor_static = foo();
    // CHECK that the static in the ctor gets mangled correctly:
    // CHECK: @"\01?ctor_static@?1???0D@basic@@QAE@XZ@4HA"
  }
  ~D() {
    static int dtor_static = foo();
    // CHECK that the static in the dtor gets mangled correctly:
    // CHECK: @"\01?dtor_static@?1???1D@basic@@QAE@XZ@4HA"
  }
};

void use_D() { D c; }

// DTORS: attributes [[NUW]] = { nounwind{{.*}} }

} // end namespace basic


namespace constructors {

struct A {
  A() {}
};

struct B : A {
  B();
  ~B();
};

B::B() {
  // CHECK: define x86_thiscallcc %"struct.constructors::B"* @"\01??0B@constructors@@QAE@XZ"(%"struct.constructors::B"* returned %this)
  // CHECK: call x86_thiscallcc %"struct.constructors::A"* @"\01??0A@constructors@@QAE@XZ"(%"struct.constructors::A"* %{{.*}})
  // CHECK: ret
}

struct C : virtual A {
  C();
};

C::C() {
  // CHECK: define x86_thiscallcc %"struct.constructors::C"* @"\01??0C@constructors@@QAE@XZ"(%"struct.constructors::C"* returned %this, i32 %is_most_derived)
  // TODO: make sure this works in the Release build too;
  // CHECK: store i32 %is_most_derived, i32* %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32* %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: bitcast %"struct.constructors::C"* %{{.*}} to %"struct.constructors::A"*
  // CHECK-NEXT: call x86_thiscallcc %"struct.constructors::A"* @"\01??0A@constructors@@QAE@XZ"(%"struct.constructors::A"* %{{.*}})
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // CHECK: @"\01??_7C@constructors@@6B@"
  // CHECK: ret
}

void create_C() {
  C c;
  // CHECK: define void @"\01?create_C@constructors@@YAXXZ"()
  // CHECK: call x86_thiscallcc %"struct.constructors::C"* @"\01??0C@constructors@@QAE@XZ"(%"struct.constructors::C"* %c, i32 1)
  // CHECK: ret
}

struct D : C {
  D();
};

D::D() {
  // CHECK: define x86_thiscallcc %"struct.constructors::D"* @"\01??0D@constructors@@QAE@XZ"(%"struct.constructors::D"* returned %this, i32 %is_most_derived) unnamed_addr
  // CHECK: store i32 %is_most_derived, i32* %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32* %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: bitcast %"struct.constructors::D"* %{{.*}} to %"struct.constructors::A"*
  // CHECK-NEXT: call x86_thiscallcc %"struct.constructors::A"* @"\01??0A@constructors@@QAE@XZ"(%"struct.constructors::A"* %{{.*}})
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // CHECK: call x86_thiscallcc %"struct.constructors::C"* @"\01??0C@constructors@@QAE@XZ"(%"struct.constructors::C"* %{{.*}}, i32 0)
  // CHECK: ret
}

struct E : virtual C {
  E();
};

E::E() {
  // CHECK: define x86_thiscallcc %"struct.constructors::E"* @"\01??0E@constructors@@QAE@XZ"(%"struct.constructors::E"* returned %this, i32 %is_most_derived) unnamed_addr
  // CHECK: store i32 %is_most_derived, i32* %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // CHECK: %[[IS_MOST_DERIVED_VAL:.*]] = load i32* %[[IS_MOST_DERIVED_VAR]]
  // CHECK: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // CHECK: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  //
  // CHECK: [[INIT_VBASES]]
  // CHECK-NEXT: bitcast %"struct.constructors::E"* %{{.*}} to %"struct.constructors::A"*
  // CHECK-NEXT: call x86_thiscallcc %"struct.constructors::A"* @"\01??0A@constructors@@QAE@XZ"(%"struct.constructors::A"* %{{.*}})
  // CHECK: call x86_thiscallcc %"struct.constructors::C"* @"\01??0C@constructors@@QAE@XZ"(%"struct.constructors::C"* %{{.*}}, i32 0)
  // CHECK-NEXT: br label %[[SKIP_VBASES]]
  //
  // CHECK: [[SKIP_VBASES]]
  // CHECK: ret
}

} // end namespace constructors

// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -o - | FileCheck %s

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=x86_64-pc-win32 -emit-llvm -o %t

struct VBase {
  virtual ~VBase();
  virtual void foo();
  virtual void bar();
  int field;
};

struct B : virtual VBase {
  B();
  virtual ~B();
  virtual void foo();
  virtual void bar();
};

B::B() {
  // CHECK-LABEL: define x86_thiscallcc %struct.B* @"\01??0B@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load %struct.B**
  // CHECK:   br i1 %{{.*}}, label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]

  // Don't check the INIT_VBASES case as it's covered by the ctor tests.

  // CHECK: %[[SKIP_VBASES]]
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 %{{.*}}
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to [3 x i8*]**
  // CHECK:   store [3 x i8*]* @"\01??_7B@@6B@", [3 x i8*]** %[[VFPTR]]

  // Initialize vtorDisp:
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8* %[[VBASE_i8]], i32 -4
  // CHECK:   %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_i8]] to i32*
  // CHECK:   store i32 %[[VTORDISP_VAL]], i32* %[[VTORDISP_PTR]]

  // CHECK: ret
}

B::~B() {
  // CHECK-LABEL: define x86_thiscallcc void @"\01??1B@@UAE@XZ"
  // CHECK:   %[[THIS:.*]] = load %struct.B**

  // Restore the vfptr that could have been changed by a subclass.
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 %{{.*}}
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to [3 x i8*]**
  // CHECK:   store [3 x i8*]* @"\01??_7B@@6B@", [3 x i8*]** %[[VFPTR]]

  // Initialize vtorDisp:
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBPTR:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 0
  // ...
  // CHECK:   %[[VBASE_OFFSET:.*]] = add nsw i32 0, %{{.*}}
  // CHECK:   %[[VTORDISP_VAL:.*]] = sub i32 %[[VBASE_OFFSET]], 8
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK:   %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i32 %[[VBASE_OFFSET]]
  // CHECK:   %[[VTORDISP_i8:.*]] = getelementptr i8* %[[VBASE_i8]], i32 -4
  // CHECK:   %[[VTORDISP_PTR:.*]] = bitcast i8* %[[VTORDISP_i8]] to i32*
  // CHECK:   store i32 %[[VTORDISP_VAL]], i32* %[[VTORDISP_PTR]]

  foo();  // Avoid the "trivial destructor" optimization.

  // CHECK: ret
}

void B::foo() {
// CHECK-LABEL: define x86_thiscallcc void @"\01?foo@B@@UAEXXZ"(i8*
//
// B::foo gets 'this' cast to VBase* in ECX (i.e. this+8) so we
// need to adjust 'this' before use.
//
// CHECK: %[[THIS_ADDR:.*]] = alloca %struct.B*, align 4
// CHECK: %[[THIS_i8:.*]] = getelementptr i8* %[[ECX:.*]], i64 -8
// CHECK: %[[THIS:.*]] = bitcast i8* %[[THIS_i8]] to %struct.B*
// CHECK: store %struct.B* %[[THIS]], %struct.B** %[[THIS_ADDR]], align 4

  field = 42;
// CHECK: %[[THIS:.*]] = load %struct.B** %[[THIS_ADDR]]
// CHECK: %[[THIS8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8* %[[THIS8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i8**
// CHECK: %[[VBTABLE:.*]] = load i8** %[[VBPTR8]]
// CHECK: %[[VBENTRY8:.*]] = getelementptr inbounds i8* %[[VBTABLE]], i32 4
// CHECK: %[[VBENTRY:.*]] = bitcast i8* %[[VBENTRY8]] to i32*
// CHECK: %[[VBOFFSET32:.*]] = load i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[THIS8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[THIS8]], i32 %[[VBOFFSET]]
// CHECK: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.VBase*
// CHECK: %[[FIELD:.*]] = getelementptr inbounds %struct.VBase* %[[VBASE]], i32 0, i32 1
// CHECK: store i32 42, i32* %[[FIELD]], align 4
//
// CHECK: ret void
}

void call_vbase_bar(B *obj) {
// CHECK-LABEL: define void @"\01?call_vbase_bar@@YAXPAUB@@@Z"(%struct.B* %obj)
// CHECK: %[[OBJ:.*]] = load %struct.B

  obj->bar();
// When calling a vbase's virtual method, one needs to adjust 'this'
// at the caller site.
//
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i8**
// CHECK: %[[VBTABLE:.*]] = load i8** %[[VBPTR8]]
// CHECK: %[[VBENTRY8:.*]] = getelementptr inbounds i8* %[[VBTABLE]], i32 4
// CHECK: %[[VBENTRY:.*]] = bitcast i8* %[[VBENTRY8]] to i32*
// CHECK: %[[VBOFFSET32:.*]] = load i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[VFPTR:.*]] = bitcast i8* %[[VBASE_i8]] to void (i8*)***
// CHECK: %[[VFTABLE:.*]] = load void (i8*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (i8*)** %[[VFTABLE]], i64 2
// CHECK: %[[VFUN_VALUE:.*]] = load void (i8*)** %[[VFUN]]
//
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i8**
// CHECK: %[[VBTABLE:.*]] = load i8** %[[VBPTR8]]
// CHECK: %[[VBENTRY8:.*]] = getelementptr inbounds i8* %[[VBTABLE]], i32 4
// CHECK: %[[VBENTRY:.*]] = bitcast i8* %[[VBENTRY8]] to i32*
// CHECK: %[[VBOFFSET32:.*]] = load i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](i8* %[[VBASE]])
//
// CHECK: ret void
}

struct C : B {
  C();
  // has an implicit vdtor.
};

// Used to crash on an assertion.
C::C() {}

namespace multiple_vbases {
struct A {
  virtual void a();
};

struct B {
  virtual void b();
};

struct C {
  virtual void c();
};

struct D : virtual A, virtual B, virtual C {
  virtual void a();
  virtual void b();
  virtual void c();
  D();
};

D::D() {
  // CHECK-LABEL: define x86_thiscallcc %"struct.multiple_vbases::D"* @"\01??0D@multiple_vbases@@QAE@XZ"
  // Just make sure we emit 3 vtordisps after initializing vfptrs.
  // CHECK: store [1 x i8*]* @"\01??_7D@multiple_vbases@@6BA@1@@", [1 x i8*]** %{{.*}}
  // CHECK: store [1 x i8*]* @"\01??_7D@multiple_vbases@@6BB@1@@", [1 x i8*]** %{{.*}}
  // CHECK: store [1 x i8*]* @"\01??_7D@multiple_vbases@@6BC@1@@", [1 x i8*]** %{{.*}}
  // ...
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
  // CHECK: store i32 %{{.*}}, i32* %{{.*}}
  // CHECK: ret
}

}

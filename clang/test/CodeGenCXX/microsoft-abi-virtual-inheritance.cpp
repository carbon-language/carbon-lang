// RUN: %clang_cc1 %s -fno-rtti -cxx-abi microsoft -triple=i386-pc-win32 -emit-llvm -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=CHECK2 %s < %t

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
  // Adjust the this parameter:
  // CHECK:   %[[THIS_PARAM_i8:.*]] = bitcast %struct.B* {{.*}} to i8*
  // CHECK:   %[[THIS_i8:.*]] = getelementptr inbounds i8* %[[THIS_PARAM_i8]], i32 -8
  // CHECK:   %[[THIS:.*]] = bitcast i8* %[[THIS_i8]] to %struct.B*
  // CHECK:   store %struct.B* %[[THIS]], %struct.B** %[[THIS_ADDR:.*]], align 4
  // CHECK:   %[[THIS:.*]] = load %struct.B** %[[THIS_ADDR]]

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

  // CHECK2-LABEL: define linkonce_odr x86_thiscallcc void @"\01??_DB@@UAE@XZ"(%struct.B*
  // CHECK2: %[[THIS:.*]] = load %struct.B** {{.*}}
  // CHECK2: %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK2: %[[B_i8:.*]] = getelementptr i8* %[[THIS_i8]], i32 8
  // CHECK2: %[[B:.*]] = bitcast i8* %[[B_i8]] to %struct.B*
  // CHECK2: call x86_thiscallcc void @"\01??1B@@UAE@XZ"(%struct.B* %[[B]])
  // CHECK2: %[[THIS_i8:.*]] = bitcast %struct.B* %[[THIS]] to i8*
  // CHECK2: %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[THIS_i8]], i64 8
  // CHECK2: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.VBase*
  // CHECK2: call x86_thiscallcc void @"\01??1VBase@@UAE@XZ"(%struct.VBase* %[[VBASE]])
  // CHECK2: ret

  // CHECK2-LABEL: define linkonce_odr x86_thiscallcc void @"\01??_GB@@UAEPAXI@Z"
  // CHECK2:   %[[THIS_PARAM_i8:.*]] = bitcast %struct.B* {{.*}} to i8*
  // CHECK2:   %[[THIS_i8:.*]] = getelementptr inbounds i8* %[[THIS_PARAM_i8:.*]], i32 -8
  // CHECK2:   %[[THIS:.*]] = bitcast i8* %[[THIS_i8]] to %struct.B*
  // CHECK2:   store %struct.B* %[[THIS]], %struct.B** %[[THIS_ADDR:.*]], align 4
  // CHECK2:   %[[THIS:.*]] = load %struct.B** %[[THIS_ADDR]]
  // CHECK2:   call x86_thiscallcc void @"\01??_DB@@UAE@XZ"(%struct.B* %[[THIS]])
  // ...
  // CHECK2: ret
}

void B::foo() {
// CHECK-LABEL: define x86_thiscallcc void @"\01?foo@B@@UAEXXZ"(i8*
//
// B::foo gets 'this' cast to VBase* in ECX (i.e. this+8) so we
// need to adjust 'this' before use.
//
// CHECK: %[[THIS_ADDR:.*]] = alloca %struct.B*, align 4
// CHECK: %[[THIS_i8:.*]] = getelementptr inbounds i8* %[[ECX:.*]], i32 -8
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

void delete_B(B *obj) {
// CHECK-LABEL: define void @"\01?delete_B@@YAXPAUB@@@Z"(%struct.B* %obj)
// CHECK: %[[OBJ:.*]] = load %struct.B

  delete obj;
// CHECK: %[[OBJ_i8:.*]] = bitcast %struct.B* %[[OBJ]] to i8*
// CHECK: %[[VBPTR:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 0
// CHECK: %[[VBPTR8:.*]] = bitcast i8* %[[VBPTR]] to i8**
// CHECK: %[[VBTABLE:.*]] = load i8** %[[VBPTR8]]
// CHECK: %[[VBENTRY8:.*]] = getelementptr inbounds i8* %[[VBTABLE]], i32 4
// CHECK: %[[VBENTRY:.*]] = bitcast i8* %[[VBENTRY8]] to i32*
// CHECK: %[[VBOFFSET32:.*]] = load i32* %[[VBENTRY]]
// CHECK: %[[VBOFFSET:.*]] = add nsw i32 0, %[[VBOFFSET32]]
// CHECK: %[[VBASE_i8:.*]] = getelementptr inbounds i8* %[[OBJ_i8]], i32 %[[VBOFFSET]]
// CHECK: %[[VFPTR:.*]] = bitcast i8* %[[VBASE_i8]] to void (%struct.B*, i32)***
// CHECK: %[[VFTABLE:.*]] = load void (%struct.B*, i32)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (%struct.B*, i32)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load void (%struct.B*, i32)** %[[VFUN]]
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
// CHECK: %[[VBASE:.*]] = bitcast i8* %[[VBASE_i8]] to %struct.B*
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](%struct.B* %[[VBASE]], i32 1)
// CHECK: ret void
}

void call_complete_dtor() {
  // CHECK-LABEL: define void @"\01?call_complete_dtor@@YAXXZ"
  B b;
  // CHECK: call x86_thiscallcc %struct.B* @"\01??0B@@QAE@XZ"(%struct.B* %[[B:.*]], i32 1)
  // CHECK-NOT: getelementptr
  // CHECK: call x86_thiscallcc void @"\01??_DB@@UAE@XZ"(%struct.B* %[[B]])
  // CHECK: ret
}

struct C : B {
  C();
  // has an implicit vdtor.
};

// Used to crash on an assertion.
C::C() {
// CHECK-LABEL: define x86_thiscallcc %struct.C* @"\01??0C@@QAE@XZ"
}

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

namespace diamond {
struct A {
  A();
  virtual ~A();
};

struct B : virtual A {
  B();
  ~B();
};

struct C : virtual A {
  C();
  ~C();
  int c1, c2, c3;
};

struct Z {
  int z;
};

struct D : virtual Z, B, C {
  D();
  ~D();
} d;

D::~D() {
  // CHECK-LABEL: define x86_thiscallcc void @"\01??1D@diamond@@UAE@XZ"(%"struct.diamond::D"*)
  // CHECK: %[[ARG_i8:.*]] = bitcast %"struct.diamond::D"* %{{.*}} to i8*
  // CHECK: %[[THIS_i8:.*]] = getelementptr inbounds i8* %[[ARG_i8]], i32 -24
  // CHECK: %[[THIS:.*]] = bitcast i8* %[[THIS_i8]] to %"struct.diamond::D"*
  // CHECK: store %"struct.diamond::D"* %[[THIS]], %"struct.diamond::D"** %[[THIS_VAL:.*]], align 4
  // CHECK: %[[THIS:.*]] = load %"struct.diamond::D"** %[[THIS_VAL]]
  // CHECK: %[[D_i8:.*]] = bitcast %"struct.diamond::D"* %[[THIS]] to i8*
  // CHECK: %[[C_i8:.*]] = getelementptr inbounds i8* %[[D_i8]], i64 4
  // CHECK: %[[C:.*]] = bitcast i8* %[[C_i8]] to %"struct.diamond::C"*
  // CHECK: %[[C_i8:.*]] = bitcast %"struct.diamond::C"* %[[C]] to i8*
  // CHECK: %[[ARG_i8:.*]] = getelementptr i8* %{{.*}}, i32 16
  // FIXME: We might consider changing the dtor this parameter type to i8*.
  // CHECK: %[[ARG:.*]] = bitcast i8* %[[ARG_i8]] to %"struct.diamond::C"*
  // CHECK: call x86_thiscallcc void @"\01??1C@diamond@@UAE@XZ"(%"struct.diamond::C"* %[[ARG]])

  // CHECK: %[[B:.*]] = bitcast %"struct.diamond::D"* %[[THIS]] to %"struct.diamond::B"*
  // CHECK: %[[B_i8:.*]] = bitcast %"struct.diamond::B"* %[[B]] to i8*
  // CHECK: %[[ARG_i8:.*]] = getelementptr i8* %[[B_i8]], i32 4
  // CHECK: %[[ARG:.*]] = bitcast i8* %[[ARG_i8]] to %"struct.diamond::B"*
  // CHECK: call x86_thiscallcc void @"\01??1B@diamond@@UAE@XZ"(%"struct.diamond::B"* %[[ARG]])
  // CHECK: ret void
}

}

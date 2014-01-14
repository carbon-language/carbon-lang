// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 >%t 2>&1
// RUN: FileCheck --check-prefix=MANGLING %s < %t
// RUN: FileCheck --check-prefix=XMANGLING %s < %t
// RUN: FileCheck --check-prefix=CODEGEN %s < %t
// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -triple=x86_64-pc-win32 2>&1 | FileCheck --check-prefix=MANGLING-X64 %s

void foo(void *);

struct A {
  virtual ~A();
  virtual void public_f();
  // Make sure we don't emit unneeded thunks:
  // XMANGLING-NOT: @"\01?public_f@A@@QAEXXZ"
 protected:
  virtual void protected_f();
 private:
  virtual void private_f();
};

struct B {
  virtual ~B();
  virtual void public_f();
 protected:
  virtual void protected_f();
 private:
  virtual void private_f();
};


struct C : A, B {
  C();

  virtual ~C();
  // MANGLING-DAG: @"\01??1C@@UAE@XZ"
  // MANGLING-DAG: @"\01??_GC@@UAEPAXI@Z"
  // MANGLING-DAG: @"\01??_EC@@W3AEPAXI@Z"
  // MANGLING-X64-DAG: @"\01??1C@@UEAA@XZ"
  // MANGLING-X64-DAG: @"\01??_GC@@UEAAPEAXI@Z"
  // MANGLING-X64-DAG: @"\01??_EC@@W7EAAPEAXI@Z"

  // Overrides public_f() of two subobjects with distinct vfptrs, thus needs a thunk.
  virtual void public_f();
  // MANGLING-DAG: @"\01?public_f@C@@UAEXXZ"
  // MANGLING-DAG: @"\01?public_f@C@@W3AEXXZ"
  // MANGLING-X64-DAG: @"\01?public_f@C@@UEAAXXZ"
  // MANGLING-X64-DAG: @"\01?public_f@C@@W7EAAXXZ"
 protected:
  virtual void protected_f();
  // MANGLING-DAG: @"\01?protected_f@C@@MAEXXZ"
  // MANGLING-DAG: @"\01?protected_f@C@@O3AEXXZ"
  // MANGLING-X64-DAG: @"\01?protected_f@C@@MEAAXXZ"
  // MANGLING-X64-DAG: @"\01?protected_f@C@@O7EAAXXZ"

 private:
  virtual void private_f();
  // MANGLING-DAG: @"\01?private_f@C@@EAEXXZ"
  // MANGLING-DAG: @"\01?private_f@C@@G3AEXXZ"
  // MANGLING-X64-DAG: @"\01?private_f@C@@EEAAXXZ"
  // MANGLING-X64-DAG: @"\01?private_f@C@@G7EAAXXZ"
};

C::C() {}  // Emits vftable and forces thunk generation.

// CODEGEN-LABEL: define weak x86_thiscallcc void @"\01??_EC@@W3AEPAXI@Z"(%struct.C* %this, i32 %should_call_delete)
// CODEGEN:   getelementptr i8* {{.*}}, i32 -4
// FIXME: should actually call _EC, not _GC.
// CODEGEN:   call x86_thiscallcc void @"\01??_GC@@UAEPAXI@Z"
// CODEGEN: ret

// CODEGEN-LABEL: define weak x86_thiscallcc void @"\01?public_f@C@@W3AEXXZ"(%struct.C*
// CODEGEN:   getelementptr i8* {{.*}}, i32 -4
// CODEGEN:   call x86_thiscallcc void @"\01?public_f@C@@UAEXXZ"(%struct.C*
// CODEGEN: ret

void zoo(C* obj) {
  delete obj;
}

struct D {
  virtual B* goo();
};

struct E : D {
  E();
  virtual C* goo();
  // MANGLING-DAG: @"\01?goo@E@@UAEPAUC@@XZ"
  // MANGLING-DAG: @"\01?goo@E@@QAEPAUB@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@E@@UEAAPEAUC@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@E@@QEAAPEAUB@@XZ"
};

E::E() {}  // Emits vftable and forces thunk generation.

// CODEGEN-LABEL: define weak x86_thiscallcc %struct.C* @"\01?goo@E@@QAEPAUB@@XZ"
// CODEGEN:   call x86_thiscallcc %struct.C* @"\01?goo@E@@UAEPAUC@@XZ"
// CODEGEN:   getelementptr inbounds i8* {{.*}}, i32 4
// CODEGEN: ret

struct F : virtual A, virtual B {
  virtual void own_method();
  virtual ~F();
};

F f;  // Just make sure we don't crash, e.g. mangling the complete dtor.

struct G : C { };

struct H : E {
  virtual G* goo();
  // MANGLING-DAG: @"\01?goo@H@@UAEPAUG@@XZ"
  // MANGLING-DAG: @"\01?goo@H@@QAEPAUB@@XZ"
  // MANGLING-DAG: @"\01?goo@H@@QAEPAUC@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@H@@UEAAPEAUG@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@H@@QEAAPEAUB@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@H@@QEAAPEAUC@@XZ"
};

H h;

struct I : D {
  I();
  virtual F* goo();
};

I::I() {}  // Emits vftable and forces thunk generation.

// CODEGEN-LABEL: define weak x86_thiscallcc %struct.{{[BF]}}* @"\01?goo@I@@QAEPAUB@@XZ"
// CODEGEN: %[[ORIG_RET:.*]] = call x86_thiscallcc %struct.F* @"\01?goo@I@@UAEPAUF@@XZ"
// CODEGEN: %[[ORIG_RET_i8:.*]] = bitcast %struct.F* %[[ORIG_RET]] to i8*
// CODEGEN: %[[VBPTR_i8:.*]] = getelementptr inbounds i8* %[[ORIG_RET_i8]], i32 4
// CODEGEN: %[[VBPTR:.*]] = bitcast i8* %[[VBPTR_i8]] to i8**
// CODEGEN: %[[VBTABLE:.*]] = load i8** %[[VBPTR]]
// CODEGEN: %[[VBASE_OFFSET_PTR_i8:.*]] = getelementptr inbounds i8* %[[VBTABLE]], i32 8
// CODEGEN: %[[VBASE_OFFSET_PTR:.*]] = bitcast i8* %[[VBASE_OFFSET_PTR_i8]] to i32*
// CODEGEN: %[[VBASE_OFFSET:.*]] = load i32* %[[VBASE_OFFSET_PTR]]
// CODEGEN: %[[RES_i8:.*]] = getelementptr inbounds i8* %[[VBPTR_i8]], i32 %[[VBASE_OFFSET]]
// CODEGEN: %[[RES:.*]] = bitcast i8* %[[RES_i8]] to %struct.F*
// CODEGEN: phi %struct.F* {{.*}} %[[RES]]
// CODEGEN: ret %struct.{{[BF]}}*

namespace CrashOnThunksForAttributedType {
// We used to crash on this because the type of foo is an AttributedType, not
// FunctionType, and we had to look through the sugar.
struct A {
  virtual void __stdcall foo();
};
struct B {
  virtual void __stdcall foo();
};
struct C : A, B {
  virtual void __stdcall foo();
};
C c;
}

// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 >%t 2>&1
// RUN: FileCheck --check-prefix=MANGLING %s < %t
// RUN: FileCheck --check-prefix=XMANGLING %s < %t
// RUN: FileCheck --check-prefix=CODEGEN %s < %t
// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -cxx-abi microsoft -triple=x86_64-pc-win32 2>&1 | FileCheck --check-prefix=MANGLING-X64 %s

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

// CODEGEN: define weak x86_thiscallcc void @"\01??_EC@@W3AEPAXI@Z"(%struct.C* %this, i32 %should_call_delete)
// CODEGEN:   getelementptr inbounds i8* {{.*}}, i64 -4
// FIXME: should actually call _EC, not _GC.
// CODEGEN:   call x86_thiscallcc void @"\01??_GC@@UAEPAXI@Z"
// CODEGEN: ret

// CODEGEN: define weak x86_thiscallcc void @"\01?public_f@C@@W3AEXXZ"(%struct.C*
// CODEGEN:   getelementptr inbounds i8* {{.*}}, i64 -4
// CODEGEN:   call x86_thiscallcc void @"\01?public_f@C@@UAEXXZ"(%struct.C*
// CODEGEN: ret

void zoo(C* obj) {
  delete obj;
}

struct D {
  virtual B* goo();
};

struct E : D {
  virtual C* goo();
  // MANGLING-DAG: @"\01?goo@E@@UAEPAUC@@XZ"
  // MANGLING-DAG: @"\01?goo@E@@QAEPAUB@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@E@@UEAAPEAUC@@XZ"
  // MANGLING-X64-DAG: @"\01?goo@E@@QEAAPEAUB@@XZ"
};

E e;  // Emits vftable and forces thunk generation.

// CODEGEN: define weak x86_thiscallcc %struct.C* @"\01?goo@E@@QAEPAUB@@XZ"
// CODEGEN:   call x86_thiscallcc %struct.C* @"\01?goo@E@@UAEPAUC@@XZ"
// CODEGEN:   getelementptr inbounds i8* {{.*}}, i64 4
// CODEGEN: ret

struct F : virtual A, virtual B {
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

// FIXME: Write vtordisp adjusting thunk tests

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

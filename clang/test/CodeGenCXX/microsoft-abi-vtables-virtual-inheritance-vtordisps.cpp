// RUN: %clang_cc1 -fno-rtti -emit-llvm -fdump-vtable-layouts %s -o %t.ll -cxx-abi microsoft -triple=i386-pc-win32 >%t
// RUN: FileCheck --check-prefix=VTABLE-SIMPLE-A %s < %t
// RUN: FileCheck --check-prefix=VTABLE-SIMPLE-B %s < %t
// RUN: FileCheck --check-prefix=VTABLE-SIMPLE-C %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-A %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-B %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-C %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-E %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-F %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-G %s < %t
// RUN: FileCheck --check-prefix=VTABLE-EXTENDED-H %s < %t
// RUN: FileCheck --check-prefix=VTABLE-PR17738-A %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -fdump-vtable-layouts %s -cxx-abi microsoft -triple=x86_64-pc-win32 >/dev/null

struct V1 {
  virtual void f();
  virtual ~V1();
};

struct V2 {
  virtual void f();
  virtual ~V2();
  int v;
};

struct Z {
  virtual void g();
  virtual ~Z();
  int x;
};

struct V3 : Z, V2 {
};

struct V4 : Z, V1, V2 {
  int y;
};

void use_somewhere_else(void*);

namespace simple {
// In case of a single-layer virtual inheritance, the "this" adjustment is done
// staically:
//   struct A {
//     virtual void f();  // Expects "(A*)this" in ECX
//   };
//   struct B : virtual A {
//     virtual void f();  // Expects "(char*)(B*)this + 12" in ECX
//     virtual ~B();      // Might call f()
//   };
//
// If a class overrides a virtual function of its base and has a non-trivial
// ctor/dtor that call(s) the virtual function (or may escape "this" to some
// code that might call it), a virtual adjustment might be needed in case the
// current class layout and the most derived class layout are different.
// This is done using vtordisp thunks.
//
// A simple vtordisp{A,B} thunk for Method@Class is something like:
//   sub  ecx, [ecx+A]  // apply the vtordisp adjustment
//   sub  ecx, B        // apply the subobject adjustment, if needed.
//   jmp Method@Class

struct A : virtual V1 {
  // VTABLE-SIMPLE-A: VFTable for 'V1' in 'simple::A' (2 entries).
  // VTABLE-SIMPLE-A-NEXT: 0 | void simple::A::f()
  // VTABLE-SIMPLE-A-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]
  // VTABLE-SIMPLE-A-NEXT: 1 | simple::A::~A() [scalar deleting]
  // VTABLE-SIMPLE-A-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  virtual void f();
  // MANGLING-DAG: @"\01?f@A@simple@@$4PPPPPPPM@A@AEXXZ"

  virtual ~A();
  // MANGLING-DAG: @"\01??_EA@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

A a;

struct B : virtual V3 {
  // VTABLE-SIMPLE-B: VFTable for 'Z' in 'V3' in 'simple::B' (2 entries).
  // VTABLE-SIMPLE-B-NEXT: 0 | void Z::g()
  // VTABLE-SIMPLE-B-NEXT: 1 | simple::B::~B() [scalar deleting]
  // VTABLE-SIMPLE-B-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // VTABLE-SIMPLE-B: VFTable for 'V2' in 'V3' in 'simple::B' (2 entries).
  // VTABLE-SIMPLE-B-NEXT: 0 | void simple::B::f()
  // VTABLE-SIMPLE-B-NEXT:     [this adjustment: vtordisp at -12, 0 non-virtual]
  // VTABLE-SIMPLE-B-NEXT: 1 | simple::B::~B() [scalar deleting]
  // VTABLE-SIMPLE-B-NEXT:     [this adjustment: vtordisp at -12, -8 non-virtual]

  // FIXME: The vtordisp thunk should only get emitted for a constructor
  // if "this" leaves scope.
  B() { use_somewhere_else(this); }

  virtual void f();
  // MANGLING-DAG: @"\01?f@B@simple@@$4PPPPPPPE@A@AEXXZ"

  // Has an implicit destructor.
  // MANGLING-DAG: @"\01??_EB@simple@@$4PPPPPPPE@7AEPAXI@Z"
  // MANGLING-DAG: @"\01??_EB@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

B b;

struct C : virtual V4 {
  // VTABLE-SIMPLE-C: VFTable for 'Z' in 'V4' in 'simple::C' (2 entries).
  // VTABLE-SIMPLE-C-NEXT: 0 | void Z::g()
  // VTABLE-SIMPLE-C-NEXT: 1 | simple::C::~C() [scalar deleting]
  // VTABLE-SIMPLE-C-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // VTABLE-SIMPLE-C: VFTable for 'V1' in 'V4' in 'simple::C' (2 entries).
  // VTABLE-SIMPLE-C-NEXT: 0 | void simple::C::f()
  // VTABLE-SIMPLE-C-NEXT:     [this adjustment: vtordisp at -12, 0 non-virtual]
  // VTABLE-SIMPLE-C-NEXT: 1 | simple::C::~C() [scalar deleting]
  // VTABLE-SIMPLE-C-NEXT:     [this adjustment: vtordisp at -12, -8 non-virtual]

  // VTABLE-SIMPLE-C: VFTable for 'V2' in 'V4' in 'simple::C' (2 entries).
  // VTABLE-SIMPLE-C-NEXT: 0 | void simple::C::f()
  // VTABLE-SIMPLE-C-NEXT:     [this adjustment: vtordisp at -16, -4 non-virtual]
  // VTABLE-SIMPLE-C-NEXT: 1 | simple::C::~C() [scalar deleting]
  // VTABLE-SIMPLE-C-NEXT:     [this adjustment: vtordisp at -16, -12 non-virtual]

  int x;
  virtual void f();
  // MANGLING-DAG: @"\01?f@C@simple@@$4PPPPPPPA@3AEXXZ"
  // MANGLING-DAG: @"\01?f@C@simple@@$4PPPPPPPE@A@AEXXZ"
  virtual ~C();
  // MANGLING-DAG: @"\01??_EC@simple@@$4PPPPPPPA@M@AEPAXI@Z"
  // MANGLING-DAG: @"\01??_EC@simple@@$4PPPPPPPE@7AEPAXI@Z"
  // MANGLING-DAG: @"\01??_EC@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

C c;
}

namespace extended {
// If a virtual function requires vtordisp adjustment and the final overrider
// is defined in another vitual base of the most derived class,
// we need to know two vbase offsets.
// In this case, we should use the extended form of vtordisp thunks, called
// vtordispex thunks.
//
// vtordispex{A,B,C,D} thunk for Method@Class is something like:
//   sub  ecx, [ecx+C]  // apply the vtordisp adjustment
//   sub  ecx, A        // jump to the vbtable of the most derived class
//   mov  eax, [ecx]    // load the vbtable address
//   add  ecx, [eax+B]  // lookup the final overrider's vbase offset
//   add  ecx, D        // apphy the subobject offset if needed
//   jmp Method@Class

struct A : virtual simple::A {
  // VTABLE-EXTENDED-A: VFTable for 'V1' in 'simple::A' in 'extended::A' (2 entries).
  // VTABLE-EXTENDED-A-NEXT: 0 | void simple::A::f()
  // VTABLE-EXTENDED-A-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // VTABLE-EXTENDED-A-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]
  // VTABLE-EXTENDED-A-NEXT: 1 | extended::A::~A() [scalar deleting]
  // VTABLE-EXTENDED-A-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // `vtordispex{8,8,4294967292,8}'
  // MANGLING-DAG: @"\01?f@A@simple@@$R477PPPPPPPM@7AEXXZ"

  virtual ~A();
  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"\01??_EA@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

A a;

struct B : virtual simple::A {
  // This class has an implicit dtor.  Vdtors don't require vtordispex thunks
  // as the most derived class always has an implicit dtor,
  // which is a final overrider.

  // VTABLE-EXTENDED-B: VFTable for 'V1' in 'simple::A' in 'extended::B' (2 entries).
  //  ...
  // VTABLE-EXTENDED-B: 1 | extended::B::~B() [scalar deleting]
  // VTABLE-EXTENDED-B-NEXT: [this adjustment: vtordisp at -4, 0 non-virtual]

  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"\01??_EB@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

B b;

struct C : virtual simple::A {
  // VTABLE-EXTENDED-C: VFTable for 'V1' in 'simple::A' in 'extended::C' (2 entries).
  // VTABLE-EXTENDED-C-NEXT: 0 | void simple::A::f()
  // VTABLE-EXTENDED-C-NEXT:     [this adjustment: vtordisp at -4, vbptr at 12 to the left,
  // VTABLE-EXTENDED-C-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // `vtordispex{12,8,4294967292,8}'
  // MANGLING-DAG: @"\01?f@A@simple@@$R4M@7PPPPPPPM@7AEXXZ"
  int x;
  virtual ~C();
  // MANGLING-DAG: @"\01??_EC@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

C c;

struct D : virtual V2 {
  virtual void f();
  virtual ~D();
  int x;
};

struct E : virtual D {
  // VTABLE-EXTENDED-E: VFTable for 'V2' in 'extended::D' in 'extended::E' (2 entries).
  // VTABLE-EXTENDED-E-NEXT: 0 | void extended::D::f()
  // VTABLE-EXTENDED-E-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // VTABLE-EXTENDED-E-NEXT:      vboffset at 8 in the vbtable, 12 non-virtual]

  // `vtordispex{8,8,4294967292,12}'
  // MANGLING-DAG: @"\01?f@D@extended@@$R477PPPPPPPM@M@AEXXZ"

  virtual ~E();
  // MANGLING-DAG: @"\01??_EE@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

E e;

struct F : virtual Z, virtual D {
  // VTABLE-EXTENDED-F: VFTable for 'V2' in 'extended::D' in 'extended::F' (2 entries).
  // VTABLE-EXTENDED-F-NEXT: 0 | void extended::D::f()
  // VTABLE-EXTENDED-F-NEXT:     [this adjustment: vtordisp at -4, vbptr at 20 to the left,
  // VTABLE-EXTENDED-F-NEXT:      vboffset at 12 in the vbtable, 12 non-virtual]

  // `vtordispex{20,12,4294967292,12}'
  // MANGLING-DAG: @"\01?f@D@extended@@$R4BE@M@PPPPPPPM@M@AEXXZ"
  int x;
  virtual ~F();
  // MANGLING-DAG: @"\01??_EF@extended@@$4PPPPPPPM@M@AEPAXI@Z"
};

F f;

struct G : virtual simple::A {
  // VTABLE-EXTENDED-G: VFTable for 'extended::G' (1 entries).
  // VTABLE-EXTENDED-G-NEXT: 0 | void extended::G::g()

  // VTABLE-EXTENDED-G: VFTable for 'V1' in 'simple::A' in 'extended::G' (2 entries).
  // VTABLE-EXTENDED-G-NEXT: 0 | void simple::A::f()
  // VTABLE-EXTENDED-G-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // VTABLE-EXTENDED-G-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]
  // VTABLE-EXTENDED-G-NEXT: 1 | extended::G::~G() [scalar deleting]
  // VTABLE-EXTENDED-G-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // Emits a G's own vfptr, thus moving the vbptr in the layout.
  virtual void g();

  virtual ~G();
  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"\01??_EG@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

G g;

struct H : Z, A {
  // VTABLE-EXTENDED-H: VFTable for 'Z' in 'extended::H' (2 entries).
  // VTABLE-EXTENDED-H-NEXT: 0 | void Z::g()
  // VTABLE-EXTENDED-H-NEXT: 1 | extended::H::~H() [scalar deleting]

  // VTABLE-EXTENDED-H: VFTable for 'V1' in 'simple::A' in 'extended::A' in 'extended::H' (2 entries).
  // VTABLE-EXTENDED-H-NEXT: 0 | void simple::A::f()
  // VTABLE-EXTENDED-H-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // VTABLE-EXTENDED-H-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // MANGLING-DAG: @"\01?f@A@simple@@$R477PPPPPPPM@7AEXXZ"
  // MANGLING-DAG: @"\01??_EH@extended@@$4PPPPPPPM@BA@AEPAXI@Z"
};

H h;
}

namespace pr17738 {
// These classes should have vtordispex thunks but MSVS CL miscompiles them.
// Just do the right thing.

struct A : virtual simple::B {
  // VTABLE-PR17738-A: VFTable for 'V2' in 'V3' in 'simple::B' in 'pr17738::A' (2 entries).
  // VTABLE-PR17738-A-NEXT: 0 | void simple::B::f()
  // VTABLE-PR17738-A-NEXT:     [this adjustment: vtordisp at -12, vbptr at 20 to the left,
  // VTABLE-PR17738-A-NEXT:      vboffset at 8 in the vbtable, 16 non-virtual]

  // MANGLING-DAG: @"\01?f@B@simple@@$R4BE@7PPPPPPPE@BA@AEXXZ"
  int a;
  virtual ~A();
};

A a;
}

namespace access {
struct A {
  virtual ~A();
protected:
  virtual void prot();
private:
  virtual void priv();
};

struct B : virtual A {
  virtual ~B();
protected:
  virtual void prot();
  // MANGLING-DAG: @"\01?prot@B@access@@$2PPPPPPPM@A@AEXXZ"
private:
  virtual void priv();
  // MANGLING-DAG: @"\01?priv@B@access@@$0PPPPPPPM@A@AEXXZ"
};

B b;

struct C : virtual B {
  virtual ~C();

  // MANGLING-DAG: @"\01?prot@B@access@@$R277PPPPPPPM@7AEXXZ"
  // MANGLING-DAG: @"\01?priv@B@access@@$R077PPPPPPPM@7AEXXZ"
};

C c;
}

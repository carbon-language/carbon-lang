// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -x c++ %s -triple i386-apple-darwin10 -O0 -fasm-blocks -emit-llvm -o - | FileCheck %s

// rdar://13645930

struct Foo {
  static int *ptr;
  static int a, b;
  int arr[4];
  struct Bar {
    static int *ptr;
    char arr[2];
  };
};

void t1() {
  Foo::ptr = (int *)0xDEADBEEF;
  Foo::Bar::ptr = (int *)0xDEADBEEF;
  __asm mov eax, Foo ::ptr
  __asm mov eax, Foo :: Bar :: ptr
  __asm mov eax, [Foo:: ptr]
  __asm mov eax, dword ptr [Foo :: ptr]
  __asm mov eax, dword ptr [Foo :: ptr]
// CHECK: @_Z2t1v
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3ptrE)
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3Bar3ptrE)
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3ptrE)
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3ptrE)
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3ptrE)
}

int gvar = 10;
void t2() {
  int lvar = 10;
  __asm mov eax, offset Foo::ptr
  __asm mov eax, offset Foo::Bar::ptr
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3ptrE)
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32** @_ZN3Foo3Bar3ptrE)
}

// CHECK: define void @_Z2t3v()
void t3() {
  __asm mov eax, LENGTH Foo::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, LENGTH Foo::Bar::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, LENGTH Foo::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, LENGTH Foo::Bar::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$2", "~{eax},~{dirflag},~{fpsr},~{flags}"()

  __asm mov eax, TYPE Foo::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, TYPE Foo::Bar::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, TYPE Foo::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, TYPE Foo::Bar::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()

  __asm mov eax, SIZE Foo::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, SIZE Foo::Bar::ptr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, SIZE Foo::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$16", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  __asm mov eax, SIZE Foo::Bar::arr
// CHECK: call void asm sideeffect inteldialect "mov eax, $$2", "~{eax},~{dirflag},~{fpsr},~{flags}"()

}

struct T4 {
  int x;
  static int y;
  void test();
};

// CHECK: define void @_ZN2T44testEv(
void T4::test() {
// CHECK: [[T0:%.*]] = alloca [[T4:%.*]]*,
// CHECK: [[THIS:%.*]] = load [[T4]]** [[T0]]
// CHECK: [[X:%.*]] = getelementptr inbounds [[T4]]* [[THIS]], i32 0, i32 0
  __asm mov eax, x;
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* [[X]])
  __asm mov y, eax;
// CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, eax", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* @_ZN2T41yE)
}

template <class T> struct T5 {
  template <class U> static T create(U);
  void run();
};
// CHECK: define void @_Z5test5v()
void test5() {
  // CHECK: [[X:%.*]] = alloca i32
  // CHECK: [[Y:%.*]] = alloca i32
  int x, y;
  __asm push y
  // CHECK: call void asm sideeffect inteldialect "push dword ptr $0", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* [[Y]])
  __asm call T5<int>::create<float>
  // CHECK: call void asm sideeffect inteldialect "call $0", "r,~{dirflag},~{fpsr},~{flags}"(i32 (float)* @_ZN2T5IiE6createIfEEiT_)
  __asm mov x, eax
  // CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, eax", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* [[X]])
}

// Just verify this doesn't emit an error.
void test6() {
  __asm {
   a:
   jmp a
  }
}

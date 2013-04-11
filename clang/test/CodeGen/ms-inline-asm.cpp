// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -x c++ %s -triple i386-apple-darwin10 -O0 -fasm-blocks -emit-llvm -o - | FileCheck %s

struct Foo {
  static int *ptr;
  static int a, b;
  struct Bar {
    static int *ptr;
  };
};

void t1() {
  Foo::ptr = (int *)0xDEADBEEF;
  Foo::Bar::ptr = (int *)0xDEADBEEF;
  __asm mov eax, Foo::ptr
  __asm mov eax, Foo::Bar::ptr
  __asm mov eax, [Foo::ptr]
  __asm mov eax, dword ptr [Foo::ptr]
  __asm mov eax, dword ptr [Foo::ptr]
// CHECK: @_Z2t1v
// CHECK: call void asm sideeffect inteldialect "mov eax, Foo::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, Foo::Bar::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, Foo::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr Foo::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr Foo::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

int gvar = 10;
void t2() {
  int lvar = 10;
  __asm mov eax, offset Foo::ptr
  __asm mov eax, offset Foo::Bar::ptr
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect "mov eax, Foo::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, Foo::Bar::ptr", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

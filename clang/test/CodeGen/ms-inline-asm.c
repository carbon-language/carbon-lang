// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -O0 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm {}
}

void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm nop
  __asm nop
  __asm nop
}

void t3() {
// CHECK: @t3
// CHECK: call void asm sideeffect inteldialect "nop\0A\09nop\0A\09nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm nop __asm nop __asm nop
}

void t4(void) {
// CHECK: @t4
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax", "~{ebx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov ecx, ebx", "~{ecx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm mov ebx, eax
  __asm mov ecx, ebx
}

void t5(void) {
// CHECK: @t5
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax\0A\09mov ecx, ebx", "~{ebx},~{ecx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm mov ebx, eax __asm mov ecx, ebx
}

void t6(void) {
  __asm int 0x2c
// CHECK: t6
// CHECK: call void asm sideeffect inteldialect "int $$0x2c", "~{dirflag},~{fpsr},~{flags}"()
}

void t7() {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
// CHECK: t7
// CHECK: call void asm sideeffect inteldialect "int $$0x2c", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
}

int t8() {
  __asm int 4 ; } comments for single-line asm
  __asm {}
  __asm int 4
  return 10;
// CHECK: t8
// CHECK: call void asm sideeffect inteldialect "int $$4", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "int $$4", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret i32 10
}

void t9() {
  __asm {
    push ebx
    mov ebx, 0x07
    pop ebx
  }
// CHECK: t9
// CHECK: call void asm sideeffect inteldialect "push ebx\0A\09mov ebx, $$0x07\0A\09pop ebx", "~{ebx},~{dirflag},~{fpsr},~{flags}"()
}

unsigned t10(void) {
  unsigned i = 1, j;
  __asm {
    mov eax, i
    mov j, eax
  }
  return j;
// CHECK: t10
// CHECK: [[I:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: [[J:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: store i32 1, i32* [[I]], align 4
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $1\0A\09mov dword ptr $0, eax", "=*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}, i32* %{{.*}})
// CHECK: [[RET:%[a-zA-Z0-9]+]] = load i32* [[J]], align 4
// CHECK: ret i32 [[RET]]
}

void t11(void) {
  __asm mov eax, 1
// CHECK: t11
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

unsigned t12(void) {
  unsigned i = 1, j, l = 1, m;
  __asm {
    mov eax, i
    mov j, eax
    mov eax, l
    mov m, eax
  }
  return j + m;
// CHECK: t12
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $2\0A\09mov dword ptr $0, eax\0A\09mov eax, dword ptr $3\0A\09mov dword ptr $1, eax", "=*m,=*m,*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}})
}

void t13() {
  char i = 1;
  short j = 2;
  __asm movzx eax, i
  __asm movzx eax, j
// CHECK: t13
// CHECK: call void asm sideeffect inteldialect "movzx eax, byte ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i8* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "movzx eax, word ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i16* %{{.*}})
}

void t14() {
  unsigned i = 1, j = 2;
  __asm {
    .if 1
    mov eax, i
    .else
    mov ebx, j
    .endif
  }
// CHECK: t14
// CHECK: call void asm sideeffect inteldialect ".if 1\0A\09mov eax, dword ptr $0\0A\09.else\0A\09mov ebx, j\0A\09.endif", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

int gvar = 10;
void t15() {
  int lvar = 10;
  __asm mov eax, lvar        ; eax = 10
  __asm mov eax, offset lvar ; eax = address of lvar
  __asm mov eax, offset gvar ; eax = address of gvar
// CHECK: t15
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* @{{.*}})
}

void t16() {
  int var = 10;
  __asm mov [eax], offset var
// CHECK: t16
// CHECK: call void asm sideeffect inteldialect "mov [eax], $0", "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t17() {
  __asm _emit 0x4A
  __asm _emit 0x43
  __asm _emit 0x4B
  __asm _EMIT 0x4B
// CHECK: t17
// CHECK:  call void asm sideeffect inteldialect ".byte 0x4A", "~{dirflag},~{fpsr},~{flags}"()
// CHECK:  call void asm sideeffect inteldialect ".byte 0x43", "~{dirflag},~{fpsr},~{flags}"()
// CHECK:  call void asm sideeffect inteldialect ".byte 0x4B", "~{dirflag},~{fpsr},~{flags}"()
// CHECK:  call void asm sideeffect inteldialect ".byte 0x4B", "~{dirflag},~{fpsr},~{flags}"()
}

struct t18_type { int a, b; };

int t18() {
  struct t18_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     mov eax, [ebx].0
     mov [ebx].4, ecx
  }
  return foo.b;
// CHECK: t18
// CHECK: call void asm sideeffect inteldialect "lea ebx, qword ptr foo\0A\09mov eax, [ebx].0\0A\09mov [ebx].4, ecx", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

int t19() {
  struct t18_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     mov eax, [ebx].foo.a
     mov [ebx].foo.b, ecx
  }
  return foo.b;
// CHECK: t19
// CHECK: call void asm sideeffect inteldialect "lea ebx, qword ptr foo\0A\09mov eax, [ebx].0\0A\09mov [ebx].4, ecx", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t20() {
  char bar;
  int foo;
  char _bar[2];
  int _foo[4];

  __asm mov eax, LENGTH foo
  __asm mov eax, LENGTH bar
  __asm mov eax, LENGTH _foo
  __asm mov eax, LENGTH _bar
// CHECK: t20
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$2", "~{eax},~{dirflag},~{fpsr},~{flags}"()

  __asm mov eax, TYPE foo
  __asm mov eax, TYPE bar
  __asm mov eax, TYPE _foo
  __asm mov eax, TYPE _bar
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()

  __asm mov eax, SIZE foo
  __asm mov eax, SIZE bar
  __asm mov eax, SIZE _foo
  __asm mov eax, SIZE _bar
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$16", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$2", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t21() {
  __asm {
    __asm push ebx
    __asm mov ebx, 0x07
    __asm pop ebx
  }
// CHECK: t21
// CHECK: call void asm sideeffect inteldialect "push ebx\0A\09mov ebx, $$0x07\0A\09pop ebx", "~{ebx},~{dirflag},~{fpsr},~{flags}"()
}

extern void t22_helper(int x);
void t22() {
  int x = 0;
  __asm {
    __asm push ebx
    __asm mov ebx, esp
  }
  t22_helper(x);
  __asm {
    __asm mov esp, ebx
    __asm pop ebx
  }
// CHECK: t22
// CHECK: call void asm sideeffect inteldialect "push ebx\0A\09mov ebx, esp", "~{ebx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void @t22_helper
// CHECK: call void asm sideeffect inteldialect "mov esp, ebx\0A\09pop ebx", "~{ebx},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

void t23() {
  __asm {
  the_label:
  }
// CHECK: t23
// CHECK: call void asm sideeffect inteldialect "the_label:", "~{dirflag},~{fpsr},~{flags}"()
}

void t24_helper(void) {}
void t24() {
  __asm call t24_helper
// CHECK: t24
// CHECK: call void asm sideeffect inteldialect "call $0", "r,~{dirflag},~{fpsr},~{flags}"(void ()* @t24_helper)
}

void t25() {
  __asm mov eax, 0ffffffffh
  __asm mov eax, 0fh
  __asm mov eax, 0a2h
  __asm mov eax, 0xa2h
  __asm mov eax, 0xa2
// CHECK: t25
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0ffffffffh", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0fh", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0a2h", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0xa2h", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0xa2", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t26() {
  __asm pushad
  __asm mov eax, 0
  __asm __emit 0fh
  __asm __emit 0a2h
  __asm __EMIT 0a2h
  __asm popad
// CHECK: t26
// CHECK: call void asm sideeffect inteldialect "pushad", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, $$0", "~{eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".byte 0fh", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".byte 0a2h", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".byte 0a2h", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "popad", "~{dirflag},~{fpsr},~{flags}"()
}

void t27() {
  __asm mov eax, fs:[0h]
// CHECK: t27
// CHECK: call void asm sideeffect inteldialect "mov eax, fs:[$$0h]", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t28() {
  __asm align 8
  __asm align 16;
  __asm align 128;
  __asm ALIGN 256;
// CHECK: t28
// CHECK: call void asm sideeffect inteldialect ".align 3", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".align 4", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".align 7", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect ".align 8", "~{dirflag},~{fpsr},~{flags}"()
}

void t29() {
  int arr[2] = {0, 0};
  int olen = 0, osize = 0, otype = 0;
  __asm mov olen, LENGTH arr
  __asm mov osize, SIZE arr
  __asm mov otype, TYPE arr
// CHECK: t29
// CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, $$2", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, $$8", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, $$4", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

int results[2] = {13, 37};
int *t30()
{
  int *res;
  __asm lea edi, results
  __asm mov res, edi
  return res;
// CHECK: t30
// CHECK: call void asm sideeffect inteldialect "lea edi, dword ptr $0", "*m,~{edi},~{dirflag},~{fpsr},~{flags}"([2 x i32]* @{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov dword ptr $0, edi", "=*m,~{dirflag},~{fpsr},~{flags}"(i32** %{{.*}})
}

void t31() {
  __asm pushad
  __asm popad
// CHECK: t31
// CHECK: call void asm sideeffect inteldialect "pushad", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "popad", "~{dirflag},~{fpsr},~{flags}"()
}

void t32() {
  int i;
  __asm mov eax, i
  __asm mov eax, dword ptr i
  __asm mov ax, word ptr i
  __asm mov al, byte ptr i
// CHECK: t32
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov ax, word ptr $0", "*m,~{ax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov al, byte ptr $0", "*m,~{al},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t33() {
  int i;
  __asm mov eax, [i]
  __asm mov eax, dword ptr [i]
  __asm mov ax, word ptr [i]
  __asm mov al, byte ptr [i]
// CHECK: t33
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov ax, word ptr $0", "*m,~{ax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
// CHECK: call void asm sideeffect inteldialect "mov al, byte ptr $0", "*m,~{al},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t34() {
  __asm prefetchnta 64[eax]
  __asm mov eax, dword ptr 4[eax]
// CHECK: t34
// CHECK: call void asm sideeffect inteldialect "prefetchnta $$64[eax]", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $$4[eax]", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t35() {
  __asm prefetchnta [eax + (200*64)]
  __asm mov eax, dword ptr [eax + (200*64)]
// CHECK: t35
// CHECK: call void asm sideeffect inteldialect "prefetchnta [eax + ($$200*$$64)]", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr [eax + ($$200*$$64)]", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t36() {
  int arr[4];
  __asm mov eax, 4[arr]
// CHECK: t36
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $$4$0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* %arr)
}

// RUN: %clang_cc1 -triple x86_64-pc-win32 -fasm-blocks -emit-llvm %s -o - | FileCheck %s
class t1 {
public:
  double a;
  void runc();
};

class t2 {
public:
  double a;
  void runc();
};

void t2::runc() {
  double num = 0;
  __asm {
      mov rax,[this]
      //CHECK: %this.addr = alloca %class.t2*
      //CHECK: call void asm sideeffect inteldialect "mov rax,qword ptr $1{{.*}}%class.t2* %this1
      mov rbx,[rax]
      mov num, rbx
	   };
}

void t1::runc() {
  double num = 0;
  __asm {
       mov rax,[this]
      //CHECK: %this.addr = alloca %class.t1*
      //CHECK: call void asm sideeffect inteldialect "mov rax,qword ptr $1{{.*}}%class.t1* %this1
        mov rbx,[rax]
        mov num, rbx
	   };
}

struct s {
  int a;
  void func() {
    __asm mov rax, [this]
    //CHECK: %this.addr = alloca %struct.s*
    //CHECK: call void asm sideeffect inteldialect "mov rax, qword ptr $0{{.*}}%struct.s* %this1
  }
} f3;

int main() {
  f3.func();
  f3.a=1;
  return 0;
}

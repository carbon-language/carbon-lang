// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -verify -fasm-blocks

#define M __asm int 0x2c
#define M2 int

void t1(void) { M }
void t2(void) { __asm int 0x2c }
void t3(void) { __asm M2 0x2c }
void t4(void) { __asm mov eax, fs:[0x10] }
void t5() {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
}
int t6() {
  __asm int 3 ; } comments for single-line asm
  __asm {}

  __asm int 4
  return 10;
}
void t7() {
  __asm {
    push ebx
    mov ebx, 0x07
    pop ebx
  }
}
void t8() {
  __asm nop __asm nop __asm nop
}
void t9() {
  __asm nop __asm nop ; __asm nop
}
int t_fail() { // expected-note {{to match this}}
  __asm 
  __asm { // expected-error 3 {{expected}} expected-note {{to match this}}

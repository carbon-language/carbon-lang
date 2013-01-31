; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o - | FileCheck %s

; C code this came from
;bool cas(float volatile *p, float *expected, float desired) {
;  bool success;
;  __asm__ __volatile__("lock; cmpxchg %[desired], %[mem]; "
;                       "mov %[expected], %[expected_out]; "
;                       "sete %[success]"
;                       : [success] "=a" (success),
;                         [expected_out] "=rm" (*expected)
;                       : [expected] "a" (*expected),
;                         [desired] "q" (desired),
;                         [mem] "m" (*p)
;                       : "memory", "cc");
;  return success;
;}

define zeroext i1 @cas(float* %p, float* %expected, float %desired) nounwind {
entry:
  %p.addr = alloca float*, align 8
  %expected.addr = alloca float*, align 8
  %desired.addr = alloca float, align 4
  %success = alloca i8, align 1
  store float* %p, float** %p.addr, align 8
  store float* %expected, float** %expected.addr, align 8
  store float %desired, float* %desired.addr, align 4
  %0 = load float** %expected.addr, align 8
  %1 = load float** %expected.addr, align 8
  %2 = load float* %1, align 4
  %3 = load float* %desired.addr, align 4
  %4 = load float** %p.addr, align 8
  %5 = call i8 asm sideeffect "lock; cmpxchg $3, $4; mov $2, $1; sete $0", "={ax},=*rm,{ax},q,*m,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(float* %0, float %2, float %3, float* %4) nounwind
  store i8 %5, i8* %success, align 1
  %6 = load i8* %success, align 1
  %tobool = trunc i8 %6 to i1
  ret i1 %tobool
}

; Make sure we're emitting a move from 
; CHECK: #APP
; CHECK-NEXT: lock;{{.*}}mov %eax,{{.*}}
; CHECK-NEXT: #NO_APP

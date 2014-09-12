; RUN: llc < %s -mcpu=pwr8 | FileCheck %s

; Generated from following C code:
;
; void foo (int result, char *addr) {
;   __asm__ __volatile__ (
;     "ld%U1%X1 %0,%1\n"
;     "cmpw %0,%0\n"
;     "bne- 1f\n"
;     "1: isync\n"
;     : "=r" (result)
;     : "m"(*addr) : "memory", "cr0");
; }

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind
; Check that we accept 'U' and 'X' constraints.
define void @foo(i32 signext %result, i8* %addr) #0 {
entry:
  %result.addr = alloca i32, align 4
  %addr.addr = alloca i8*, align 8
  store i32 %result, i32* %result.addr, align 4
  store i8* %addr, i8** %addr.addr, align 8
  %0 = load i8** %addr.addr, align 8
  %1 = call i32 asm sideeffect "ld${1:U}${1:X} $0,$1\0Acmpw $0,$0\0Abne- 1f\0A1: isync\0A", "=r,*m,~{memory},~{cr0}"(i8* %0) #1, !srcloc !1
  store i32 %1, i32* %result.addr, align 4
  ret void
}

; CHECK-LABEL: @foo
; CHECK: ld [[REG:[0-9]+]],0(4)
; CHECK-NEXT: cmpw [[REG]],[[REG]]
; CHECK-NEXT: bne- 1f
; CHECK-NEXT: 1: isync

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.6.0 (trunk 217557)"}
!1 = metadata !{i32 67, i32 91, i32 110, i32 126}

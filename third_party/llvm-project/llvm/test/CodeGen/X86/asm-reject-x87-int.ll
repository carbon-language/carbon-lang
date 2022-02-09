; RUN: not llc -o /dev/null %s -mtriple=i386-unknown-unknown 2>&1 | FileCheck %s

; This test was derived from this C code. The frontend sees that the constraint
; doesn't accept memory, but the argument is a strict. So it tries to bitcast
; to an integer of the same size. SelectionDAGBuilder doesn't know how to copy
; between integers and fp80 so it asserts or crashes.
;
; gcc accepts the code. But rejects it if the struct is replaced by an int. From
; the InlineAsm block those two cases look the same in LLVM IR. So if the single
; elementstruct case is valid, then the frontend needs to emit different IR.

; typedef struct float4 {
;   float f;
; } float4;
;
; int main() {
;   float4 f4;
;   f4.f = 4.0f;
;   __asm  ("fadd %%st(0), %%st(0)" : "+t" (f4));
;   return 0;
; }

%struct.float4 = type { float }

; CHECK: error: couldn't allocate output register for constraint '{st}'
define dso_local i32 @foo() {
entry:
  %retval = alloca i32, align 4
  %f4 = alloca %struct.float4, align 4
  store i32 0, i32* %retval, align 4
  %f = getelementptr inbounds %struct.float4, %struct.float4* %f4, i32 0, i32 0
  store float 4.000000e+00, float* %f, align 4
  %0 = bitcast %struct.float4* %f4 to i32*
  %1 = load i32, i32* %0, align 4
  %2 = call i32 asm "fadd %st(0), %st(0)", "={st},0,~{dirflag},~{fpsr},~{flags}"(i32 %1)
  %3 = bitcast %struct.float4* %f4 to i32*
  store i32 %2, i32* %3, align 4
  ret i32 0
}

; RUN: llc < %s -O0 -relocation-model=pic -disable-fp-elim
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

%struct0 = type { i32, i32 }

; This function would crash RegAllocFast because it tried to spill %CPSR.
define arm_apcscc void @clobber_cc() nounwind noinline ssp {
entry:
  %asmtmp = call %struct0 asm sideeffect "...", "=&r,=&r,r,Ir,r,~{cc},~{memory}"(i32* undef, i32 undef, i32 1) nounwind ; <%0> [#uses=0]
  unreachable
}

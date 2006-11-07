; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | not grep adc

; PR987

declare void %llvm.memcpy.i64(sbyte*, sbyte*, ulong, uint)

void %foo(ulong %a) {
  %b = add ulong %a, 1
call void %llvm.memcpy.i64( sbyte* null, sbyte* null, ulong %b, uint 1 )
  ret void
}

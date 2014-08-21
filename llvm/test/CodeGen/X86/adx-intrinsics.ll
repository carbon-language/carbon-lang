; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=broadwell --show-mc-encoding| FileCheck %s --check-prefix=ADX --check-prefix=CHECK

declare i8 @llvm.x86.addcarryx.u32(i8, i32, i32, i8*)

define i8 @test_addcarryx_u32(i8 %c, i32 %a, i32 %b, i8* %ptr) {
; CHECK-LABEL: test_addcarryx_u32
; CHECK: addb
; ADX: adcxl
; CHECK: setb
; CHECK: retq
  %ret = tail call i8 @llvm.x86.addcarryx.u32(i8 %c, i32 %a, i32 %b, i8* %ptr)
  ret i8 %ret;
}

declare i8 @llvm.x86.addcarryx.u64(i8, i64, i64, i8*)

define i8 @test_addcarryx_u64(i8 %c, i64 %a, i64 %b, i8* %ptr) {
; CHECK-LABEL: test_addcarryx_u64
; CHECK: addb
; ADX: adcxq
; CHECK: setb
; CHECK: retq
  %ret = tail call i8 @llvm.x86.addcarryx.u64(i8 %c, i64 %a, i64 %b, i8* %ptr)
  ret i8 %ret;
}


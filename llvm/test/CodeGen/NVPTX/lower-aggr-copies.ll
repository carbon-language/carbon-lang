; RUN: llc < %s -march=nvptx -mcpu=sm_35 | FileCheck %s

; Verify that the NVPTXLowerAggrCopies pass works as expected - calls to
; llvm.mem* intrinsics get lowered to loops.

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

define i8* @memcpy_caller(i8* %dst, i8* %src, i64 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %n, i32 1, i1 false)
  ret i8* %dst
; CHECK-LABEL: .visible .func (.param .b32 func_retval0) memcpy_caller
; CHECK: LBB[[LABEL:[_0-9]+]]:
; CHECK:      ld.u8 %rs[[REG:[0-9]+]]
; CHECK:      st.u8 [%r{{[0-9]+}}], %rs[[REG]]
; CHECK:      add.s64 %rd[[COUNTER:[0-9]+]], %rd[[COUNTER]], 1
; CHECK-NEXT: setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; CHECK-NEXT: @%p[[PRED]] bra LBB[[LABEL]]
}

define i8* @memcpy_volatile_caller(i8* %dst, i8* %src, i64 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %n, i32 1, i1 true)
  ret i8* %dst
; CHECK-LABEL: .visible .func (.param .b32 func_retval0) memcpy_volatile_caller
; CHECK: LBB[[LABEL:[_0-9]+]]:
; CHECK:      ld.volatile.u8 %rs[[REG:[0-9]+]]
; CHECK:      st.volatile.u8 [%r{{[0-9]+}}], %rs[[REG]]
; CHECK:      add.s64 %rd[[COUNTER:[0-9]+]], %rd[[COUNTER]], 1
; CHECK-NEXT: setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; CHECK-NEXT: @%p[[PRED]] bra LBB[[LABEL]]
}

define i8* @memset_caller(i8* %dst, i32 %c, i64 %n) #0 {
entry:
  %0 = trunc i32 %c to i8
  tail call void @llvm.memset.p0i8.i64(i8* %dst, i8 %0, i64 %n, i32 1, i1 false)
  ret i8* %dst
; CHECK-LABEL: .visible .func (.param .b32 func_retval0) memset_caller(
; CHECK:      ld.param.u8 %rs[[REG:[0-9]+]]
; CHECK:      LBB[[LABEL:[_0-9]+]]:
; CHECK:      st.u8 [%r{{[0-9]+}}], %rs[[REG]]
; CHECK:      add.s64 %rd[[COUNTER:[0-9]+]], %rd[[COUNTER]], 1
; CHECK-NEXT: setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; CHECK-NEXT: @%p[[PRED]] bra LBB[[LABEL]]
}

; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

define void @t0(i32 %a) nounwind {
entry:
; CHECK: t0
; CHECK: str {{w[0-9]+}}, [sp, #12]
; CHECK-NEXT: ldr [[REGISTER:w[0-9]+]], [sp, #12]
; CHECK-NEXT: str [[REGISTER]], [sp, #12]
; CHECK: ret
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr
  %tmp = load i32* %a.addr
  store i32 %tmp, i32* %a.addr
  ret void
}

define void @t1(i64 %a) nounwind {
; CHECK: t1
; CHECK: str {{x[0-9]+}}, [sp, #8]
; CHECK-NEXT: ldr [[REGISTER:x[0-9]+]], [sp, #8]
; CHECK-NEXT: str [[REGISTER]], [sp, #8]
; CHECK: ret
  %a.addr = alloca i64, align 4
  store i64 %a, i64* %a.addr
  %tmp = load i64* %a.addr
  store i64 %tmp, i64* %a.addr
  ret void
}

define zeroext i1 @i1(i1 %a) nounwind {
entry:
; CHECK: @i1
; CHECK: and w0, w0, #0x1
; CHECK: strb w0, [sp, #15]
; CHECK: ldrb w0, [sp, #15]
; CHECK: and w0, w0, #0x1
; CHECK: and w0, w0, #0x1
; CHECK: add sp, sp, #16
; CHECK: ret
  %a.addr = alloca i1, align 1
  store i1 %a, i1* %a.addr, align 1
  %0 = load i1* %a.addr, align 1
  ret i1 %0
}

define i32 @t2(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: ldur w0, [x0, #-4]
; CHECK: ret
  %0 = getelementptr i32 *%ptr, i32 -1
  %1 = load i32* %0, align 4
  ret i32 %1
}

define i32 @t3(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: ldur w0, [x0, #-256]
; CHECK: ret
  %0 = getelementptr i32 *%ptr, i32 -64
  %1 = load i32* %0, align 4
  ret i32 %1
}

define void @t4(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: movz w8, #0
; CHECK: stur w8, [x0, #-4]
; CHECK: ret
  %0 = getelementptr i32 *%ptr, i32 -1
  store i32 0, i32* %0, align 4
  ret void
}

define void @t5(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: movz w8, #0
; CHECK: stur w8, [x0, #-256]
; CHECK: ret
  %0 = getelementptr i32 *%ptr, i32 -64
  store i32 0, i32* %0, align 4
  ret void
}

define void @t6() nounwind {
; CHECK: t6
; CHECK: brk #0x1
  tail call void @llvm.trap()
  ret void
}

declare void @llvm.trap() nounwind

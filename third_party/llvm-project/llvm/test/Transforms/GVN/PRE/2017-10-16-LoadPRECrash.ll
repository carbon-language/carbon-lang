; RUN: opt -S -gvn -enable-load-pre < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%ArrayImpl = type { i64, i64 addrspace(100)*, [1 x i64], [1 x i64], [1 x i64], i64, i64, double addrspace(100)*, double addrspace(100)*, i8, i64 }

; Function Attrs: readnone
declare %ArrayImpl* @getaddr_ArrayImpl(%ArrayImpl addrspace(100)*) #0

; Function Attrs: readnone
declare i64* @getaddr_i64(i64 addrspace(100)*) #0

; Make sure that the test compiles without a crash.
; Bug https://bugs.llvm.org/show_bug.cgi?id=34937

define hidden void @wrapon_fn173() {

; CHECK-LABEL: @wrapon_fn173
; CHECK:       entry:
; CHECK-NEXT:    call %ArrayImpl* @getaddr_ArrayImpl(%ArrayImpl addrspace(100)* undef)
; CHECK-NEXT:    %.pre = load i64 addrspace(100)*, i64 addrspace(100)** null, align 8
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    call i64* @getaddr_i64(i64 addrspace(100)* %.pre)
; CHECK-NEXT:    br label %loop

entry:
  %0 = call %ArrayImpl* @getaddr_ArrayImpl(%ArrayImpl addrspace(100)* undef)
  br label %loop

loop:
  %1 = call %ArrayImpl* @getaddr_ArrayImpl(%ArrayImpl addrspace(100)* undef)
  %2 = load i64 addrspace(100)*, i64 addrspace(100)** null, align 8
  %3 = call i64* @getaddr_i64(i64 addrspace(100)* %2)
  br label %loop
}

attributes #0 = { readnone }

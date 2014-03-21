; RUN: llc < %s -march=r600 -mcpu=SI  | FileCheck %s

; In this test both the pointer and the offset operands to the
; BUFFER_LOAD instructions end up being stored in vgprs.  This
; requires us to add the pointer and offset together, store the
; result in the offset operand (vaddr), and then store 0 in an
; sgpr register pair and use that for the pointer operand
; (low 64-bits of srsrc).

; CHECK-LABEL: @mubuf
; Make sure we aren't using VGPRs for the source operand of S_MOV_B64
; CHECK-NOT: S_MOV_B64 s[{{[0-9]+:[0-9]+}}], v
define void @mubuf(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #1
  %1 = call i32 @llvm.r600.read.tidig.y() #1
  %2 = sext i32 %0 to i64
  %3 = sext i32 %1 to i64
  br label %loop

loop:
  %4 = phi i64 [0, %entry], [%5, %loop]
  %5 = add i64 %2, %4
  %6 = getelementptr i8 addrspace(1)* %in, i64 %5
  %7 = load i8 addrspace(1)* %6, align 1
  %8 = or i64 %5, 1
  %9 = getelementptr i8 addrspace(1)* %in, i64 %8
  %10 = load i8 addrspace(1)* %9, align 1
  %11 = add i8 %7, %10
  %12 = sext i8 %11 to i32
  store i32 %12, i32 addrspace(1)* %out
  %13 = icmp slt i64 %5, 10
  br i1 %13, label %loop, label %done

done:
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #1
declare i32 @llvm.r600.read.tidig.y() #1

attributes #1 = { nounwind readnone }

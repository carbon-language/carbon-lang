; RUN: llc -march=mipsel < %s | FileCheck %s

define i8* @f1() nounwind {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    or    $2, $ra, $zero
}

define i8* @f2() nounwind {
entry:
  call void @g()
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    or    $[[R0:[0-9]+]], $ra, $zero
; CHECK:    jal
; CHECK:    or    $2, $[[R0]], $zero
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
declare void @g()

; RUN: llvm-as %s -f -o %t.bc
; RUN: lli -force-interpreter=true %t.bc | tee %t.out | grep 10

; Test that APInt shift left works when bitwidth > 64 and shiftamt == 0

declare i32 @putchar(i32)

define void @putBit(i65 %x, i65 %bitnum) {
  %tmp1 = shl i65 1, %bitnum
  %tmp2 = and i65 %x, %tmp1
  %cond = icmp ne i65 %tmp2, 0
  br i1 %cond, label %cond_true, label %cond_false

cond_true:
  call i32 @putchar(i32 49)
  br label %cond_next

cond_false:
  call i32 @putchar(i32 48)
  br label %cond_next

cond_next:
  ret void
}

define i32 @main() {
  call void @putBit(i65 1, i65 0)
  call void @putBit(i65 0, i65 0)
  ret i32 0
}

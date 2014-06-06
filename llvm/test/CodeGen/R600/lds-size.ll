; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test makes sure we do not double count global values when they are
; used in different basic blocks.

; CHECK-LABEL: @test
; CHECK: .long   166120
; CHECK-NEXT: .long   1
@lds = internal unnamed_addr addrspace(3) global i32 zeroinitializer, align 4

define void @test(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %if, label %else

if:
  store i32 1, i32 addrspace(3)* @lds
  br label %endif

else:
  store i32 2, i32 addrspace(3)* @lds
  br label %endif

endif:
  ret void
}

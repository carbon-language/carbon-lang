; RUN: llc -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-apple-macosx -filetype=obj -o - \
; RUN:    | llvm-objdump -triple x86_64-apple-macosx -fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-unknown-linux-gnu -filetype=obj -o - \
; RUN:    | llvm-objdump -triple x86_64-unknown-linux-gnu -fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

define i32 @imp_null_check_load(i32* %x) {
; CHECK-LABEL: _imp_null_check_load:
; CHECK: Ltmp1:
; CHECK: movl (%rdi), %eax
; CHECK: retq
; CHECK: Ltmp0:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  ret i32 %t
}

define i32 @imp_null_check_gep_load(i32* %x) {
; CHECK-LABEL: _imp_null_check_gep_load:
; CHECK: Ltmp3:
; CHECK: movl 128(%rdi), %eax
; CHECK: retq
; CHECK: Ltmp2:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.gep = getelementptr i32, i32* %x, i32 32
  %t = load i32, i32* %x.gep
  ret i32 %t
}

define i32 @imp_null_check_add_result(i32* %x, i32 %p) {
; CHECK-LABEL: _imp_null_check_add_result:
; CHECK: Ltmp5:
; CHECK: addl (%rdi), %esi
; CHECK: movl %esi, %eax
; CHECK: retq
; CHECK: Ltmp4:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = add i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_hoist_over_unrelated_load(i32* %x, i32* %y, i32* %z) {
; CHECK-LABEL: _imp_null_check_hoist_over_unrelated_load:
; CHECK: Ltmp7:
; CHECK: movl (%rdi), %eax
; CHECK: movl (%rsi), %ecx
; CHECK: movl %ecx, (%rdx)
; CHECK: retq
; CHECK: Ltmp6:
; CHECK: movl	$42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load i32, i32* %y
  %t1 = load i32, i32* %x
  store i32 %t0, i32* %z
  ret i32 %t1
}

!0 = !{}

; CHECK-LABEL: __LLVM_FaultMaps:

; Version:
; CHECK-NEXT: .byte 1

; Reserved x2
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 0

; # functions:
; CHECK-NEXT: .long 4

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_add_result
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long Ltmp5-_imp_null_check_add_result
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long Ltmp4-_imp_null_check_add_result

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_gep_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long Ltmp3-_imp_null_check_gep_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long Ltmp2-_imp_null_check_gep_load

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_hoist_over_unrelated_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long Ltmp7-_imp_null_check_hoist_over_unrelated_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long Ltmp6-_imp_null_check_hoist_over_unrelated_load

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long Ltmp1-_imp_null_check_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long Ltmp0-_imp_null_check_load

; OBJDUMP: FaultMap table:
; OBJDUMP-NEXT: Version: 0x1
; OBJDUMP-NEXT: NumFunctions: 4
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 5
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 7
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 7
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 3

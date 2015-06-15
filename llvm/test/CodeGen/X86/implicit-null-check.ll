; RUN: llc -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

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
  br i1 %c, label %is_null, label %not_null

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
  br i1 %c, label %is_null, label %not_null

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
  br i1 %c, label %is_null, label %not_null

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = add i32 %t, %p
  ret i32 %p1
}

; CHECK-LABEL: __LLVM_FaultMaps:

; Version:
; CHECK-NEXT: .byte 1

; Reserved x2
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 0

; # functions:
; CHECK-NEXT: .long 3

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

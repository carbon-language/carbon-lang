; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

define i32 @imp_null_check_load(i32* %x) {
; CHECK-LABEL: imp_null_check_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp0:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB0_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB0_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  ret i32 %t
}

; TODO: can make implicit
define i32 @imp_null_check_unordered_load(i32* %x) {
; CHECK-LABEL: imp_null_check_unordered_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp1:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB1_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB1_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load atomic i32, i32* %x unordered, align 4
  ret i32 %t
}


; TODO: Can be converted into implicit check.
;; Probably could be implicit, but we're conservative for now
define i32 @imp_null_check_seq_cst_load(i32* %x) {
; CHECK-LABEL: imp_null_check_seq_cst_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB2_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB2_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load atomic i32, i32* %x seq_cst, align 4
  ret i32 %t
}

;; Might be memory mapped IO, so can't rely on fault behavior
define i32 @imp_null_check_volatile_load(i32* %x) {
; CHECK-LABEL: imp_null_check_volatile_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB3_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB3_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load volatile i32, i32* %x, align 4
  ret i32 %t
}


define i8 @imp_null_check_load_i8(i8* %x) {
; CHECK-LABEL: imp_null_check_load_i8:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp2:
; CHECK-NEXT:    movb (%rdi), %al ## on-fault: LBB4_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB4_1: ## %is_null
; CHECK-NEXT:    movb $42, %al
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i8* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i8 42

 not_null:
  %t = load i8, i8* %x
  ret i8 %t
}

define i256 @imp_null_check_load_i256(i256* %x) {
; CHECK-LABEL: imp_null_check_load_i256:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:  Ltmp3:
; CHECK-NEXT:    movq (%rsi), %rcx ## on-fault: LBB5_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movq 8(%rsi), %rdx
; CHECK-NEXT:    movq 16(%rsi), %rdi
; CHECK-NEXT:    movq 24(%rsi), %rsi
; CHECK-NEXT:    movq %rsi, 24(%rax)
; CHECK-NEXT:    movq %rdi, 16(%rax)
; CHECK-NEXT:    movq %rdx, 8(%rax)
; CHECK-NEXT:    movq %rcx, (%rax)
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB5_1: ## %is_null
; CHECK-NEXT:    movq $0, 24(%rax)
; CHECK-NEXT:    movq $0, 16(%rax)
; CHECK-NEXT:    movq $0, 8(%rax)
; CHECK-NEXT:    movq $42, (%rax)
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i256* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i256 42

 not_null:
  %t = load i256, i256* %x
  ret i256 %t
}



define i32 @imp_null_check_gep_load(i32* %x) {
; CHECK-LABEL: imp_null_check_gep_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp4:
; CHECK-NEXT:    movl 128(%rdi), %eax ## on-fault: LBB6_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB6_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

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
; CHECK-LABEL: imp_null_check_add_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp5:
; CHECK-NEXT:    addl (%rdi), %esi ## on-fault: LBB7_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB7_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

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

define i32 @imp_null_check_sub_result(i32* %x, i32 %p) {
; CHECK-LABEL: imp_null_check_sub_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp6:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB8_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    subl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB8_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = sub i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_mul_result(i32* %x, i32 %p) {
; CHECK-LABEL: imp_null_check_mul_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp7:
; CHECK-NEXT:    imull (%rdi), %esi ## on-fault: LBB9_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB9_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = mul i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_udiv_result(i32* %x, i32 %p) {
; CHECK-LABEL: imp_null_check_udiv_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp8:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB10_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divl %esi
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB10_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = udiv i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_shl_result(i32* %x, i32 %p) {
; CHECK-LABEL: imp_null_check_shl_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp9:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB11_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %ecx
; CHECK-NEXT:    shll %cl, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB11_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = shl i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_lshr_result(i32* %x, i32 %p) {
; CHECK-LABEL: imp_null_check_lshr_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp10:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB12_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %ecx
; CHECK-NEXT:    shrl %cl, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB12_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = lshr i32 %t, %p
  ret i32 %p1
}




define i32 @imp_null_check_hoist_over_unrelated_load(i32* %x, i32* %y, i32* %z) {
; CHECK-LABEL: imp_null_check_hoist_over_unrelated_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp11:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB13_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rsi), %ecx
; CHECK-NEXT:    movl %ecx, (%rdx)
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB13_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

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

define i32 @imp_null_check_via_mem_comparision(i32* %x, i32 %val) {
; CHECK-LABEL: imp_null_check_via_mem_comparision:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp12:
; CHECK-NEXT:    cmpl %esi, 4(%rdi) ## on-fault: LBB14_3
; CHECK-NEXT:  ## %bb.1: ## %not_null
; CHECK-NEXT:    jge LBB14_2
; CHECK-NEXT:  ## %bb.4: ## %ret_100
; CHECK-NEXT:    movl $100, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB14_3: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB14_2: ## %ret_200
; CHECK-NEXT:    movl $200, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.loc = getelementptr i32, i32* %x, i32 1
  %t = load i32, i32* %x.loc
  %m = icmp slt i32 %t, %val
  br i1 %m, label %ret_100, label %ret_200

 ret_100:
  ret i32 100

 ret_200:
  ret i32 200
}

define i32 @imp_null_check_gep_load_with_use_dep(i32* %x, i32 %a) {
; CHECK-LABEL: imp_null_check_gep_load_with_use_dep:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    ## kill: def $esi killed $esi def $rsi
; CHECK-NEXT:  Ltmp13:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB15_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    addl %edi, %esi
; CHECK-NEXT:    leal 4(%rax,%rsi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB15_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.loc = getelementptr i32, i32* %x, i32 1
  %y = ptrtoint i32* %x.loc to i32
  %b = add i32 %a, %y
  %t = load i32, i32* %x
  %z = add i32 %t, %b
  ret i32 %z
}

;; TODO: We could handle this case as we can lift the fence into the
;; previous block before the conditional without changing behavior.
define i32 @imp_null_check_load_fence1(i32* %x) {
; CHECK-LABEL: imp_null_check_load_fence1:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB16_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    ##MEMBARRIER
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB16_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

is_null:
  ret i32 42

not_null:
  fence acquire
  %t = load i32, i32* %x
  ret i32 %t
}

;; TODO: We could handle this case as we can lift the fence into the
;; previous block before the conditional without changing behavior.
define i32 @imp_null_check_load_fence2(i32* %x) {
; CHECK-LABEL: imp_null_check_load_fence2:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB17_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    mfence
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB17_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

is_null:
  ret i32 42

not_null:
  fence seq_cst
  %t = load i32, i32* %x
  ret i32 %t
}

define void @imp_null_check_store(i32* %x) {
; CHECK-LABEL: imp_null_check_store:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp14:
; CHECK-NEXT:    movl $1, (%rdi) ## on-fault: LBB18_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB18_1: ## %is_null
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret void

 not_null:
  store i32 1, i32* %x
  ret void
}

;; TODO: can be implicit
define void @imp_null_check_unordered_store(i32* %x) {
; CHECK-LABEL: imp_null_check_unordered_store:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp15:
; CHECK-NEXT:    movl $1, (%rdi) ## on-fault: LBB19_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB19_1: ## %is_null
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret void

 not_null:
  store atomic i32 1, i32* %x unordered, align 4
  ret void
}

define i32 @imp_null_check_neg_gep_load(i32* %x) {
; CHECK-LABEL: imp_null_check_neg_gep_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp16:
; CHECK-NEXT:    movl -128(%rdi), %eax ## on-fault: LBB20_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB20_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.gep = getelementptr i32, i32* %x, i32 -32
  %t = load i32, i32* %x.gep
  ret i32 %t
}

; This redefines the null check reg by doing a zero-extend and a shift on
; itself.
; Converted into implicit null check since both of these operations do not
; change the nullness of %x (i.e. if it is null, it remains null).
define i64 @imp_null_check_load_shift_addr(i64* %x) {
; CHECK-LABEL: imp_null_check_load_shift_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    shlq $6, %rdi
; CHECK-NEXT:  Ltmp17:
; CHECK-NEXT:    movq 8(%rdi), %rax ## on-fault: LBB21_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB21_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq i64* %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint i64* %x to i64
   %shry = shl i64 %y, 6
   %y.ptr = inttoptr i64 %shry to i64*
   %x.loc = getelementptr i64, i64* %y.ptr, i64 1
   %t = load i64, i64* %x.loc
   ret i64 %t
}

; Same as imp_null_check_load_shift_addr but shift is by 3 and this is now
; converted into complex addressing.
define i64 @imp_null_check_load_shift_by_3_addr(i64* %x) {
; CHECK-LABEL: imp_null_check_load_shift_by_3_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp18:
; CHECK-NEXT:    movq 8(,%rdi,8), %rax ## on-fault: LBB22_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB22_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq i64* %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint i64* %x to i64
   %shry = shl i64 %y, 3
   %y.ptr = inttoptr i64 %shry to i64*
   %x.loc = getelementptr i64, i64* %y.ptr, i64 1
   %t = load i64, i64* %x.loc
   ret i64 %t
}

define i64 @imp_null_check_load_shift_add_addr(i64* %x) {
; CHECK-LABEL: imp_null_check_load_shift_add_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK:         movq 3526(,%rdi,8), %rax ## on-fault: LBB23_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB23_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq i64* %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint i64* %x to i64
   %shry = shl i64 %y, 3
   %shry.add = add i64 %shry, 3518
   %y.ptr = inttoptr i64 %shry.add to i64*
   %x.loc = getelementptr i64, i64* %y.ptr, i64 1
   %t = load i64, i64* %x.loc
   ret i64 %t
}
!0 = !{}

; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-darwin -regalloc=basic | FileCheck %s

@__bar = external hidden global i8*
@__baz = external hidden global i8*

; rdar://8819685
define i8* @_foo() {
entry:
; CHECK-LABEL: foo:

	%size = alloca i32, align 4
	%0 = load i8** @__bar, align 4
	%1 = icmp eq i8* %0, null
	br i1 %1, label %bb1, label %bb3
; CHECK: bne
		
bb1:
	store i32 1026, i32* %size, align 4
	%2 = alloca [1026 x i8], align 1
; CHECK: mov     [[R0:r[0-9]+]], sp
; CHECK: adds    {{r[0-9]+}}, [[R0]], {{r[0-9]+}}
	%3 = getelementptr inbounds [1026 x i8]* %2, i32 0, i32 0
	%4 = call i32 @_called_func(i8* %3, i32* %size) nounwind
	%5 = icmp eq i32 %4, 0
	br i1 %5, label %bb2, label %bb3
	
bb2:
	%6 = call i8* @strdup(i8* %3) nounwind
	store i8* %6, i8** @__baz, align 4
	br label %bb3
	
bb3:
	%.0 = phi i8* [ %0, %entry ], [ %6, %bb2 ], [ %3, %bb1 ]
; CHECK: subs    r4, #5
; CHECK-NEXT: mov     sp, r4
; CHECK-NEXT: pop     {r4, r5, r6, r7, pc}
	ret i8* %.0
}

declare noalias i8* @strdup(i8* nocapture) nounwind
declare i32 @_called_func(i8*, i32*) nounwind

; Variable ending up at unaligned offset from sp (i.e. not a multiple of 4)
define void @test_local_var_addr() {
; CHECK-LABEL: test_local_var_addr:

  %addr1 = alloca i8
  %addr2 = alloca i8

; CHECK: mov r0, sp
; CHECK: adds r0, r0, #{{[0-9]+}}
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr1)

; CHECK: mov r0, sp
; CHECK: adds r0, r0, #{{[0-9]+}}
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr2)

  ret void
}

; Simple variable ending up *at* sp.
define void @test_simple_var() {
; CHECK-LABEL: test_simple_var:

  %addr32 = alloca i32
  %addr8 = bitcast i32* %addr32 to i8*

; CHECK: mov r0, sp
; CHECK-NOT: adds r0
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr8)
  ret void
}

; Simple variable ending up at aligned offset from sp.
define void @test_local_var_addr_aligned() {
; CHECK-LABEL: test_local_var_addr_aligned:

  %addr1.32 = alloca i32
  %addr1 = bitcast i32* %addr1.32 to i8*
  %addr2.32 = alloca i32
  %addr2 = bitcast i32* %addr2.32 to i8*

; CHECK: add r0, sp, #{{[0-9]+}}
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr1)

; CHECK: mov r0, sp
; CHECK-NOT: add r0
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr2)

  ret void
}

; Simple variable ending up at aligned offset from sp.
define void @test_local_var_big_offset() {
; CHECK-LABEL: test_local_var_big_offset:
  %addr1.32 = alloca i32, i32 257
  %addr1 = bitcast i32* %addr1.32 to i8*
  %addr2.32 = alloca i32, i32 257

; CHECK: add [[RTMP:r[0-9]+]], sp, #1020
; CHECL: add r0, [[RTMP]], #8
; CHECK: blx _take_ptr
  call void @take_ptr(i8* %addr1)

  ret void
}

declare void @take_ptr(i8*)

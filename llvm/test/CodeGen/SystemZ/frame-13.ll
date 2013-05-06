; Test the handling of base + 12-bit displacement addresses for large frames,
; in cases where no 20-bit form exists.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -check-prefix=CHECK-NOFP %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s

; This file tests what happens when a displacement is converted from
; being relative to the start of a frame object to being relative to
; the frame itself.  In some cases the test is only possible if two
; objects are allocated.
;
; Rather than rely on a particular order for those objects, the tests
; instead allocate two objects of the same size and apply the test to
; both of them.  For consistency, all tests follow this model, even if
; one object would actually be enough.

; First check the highest in-range offset after conversion, which is 4092
; for word-addressing instructions like MVHI.
;
; The last in-range doubleword offset is 4088.  Since the frame has an
; emergency spill slot at 160(%r15), the amount that we need to allocate
; in order to put another object at offset 4088 is (4088 - 168) / 4 = 980
; words.
define void @f1() {
; CHECK-NOFP: f1:
; CHECK-NOFP: mvhi 4092(%r15), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f1:
; CHECK-FP: mvhi 4092(%r11), 42
; CHECK-FP: br %r14
  %region1 = alloca [980 x i32], align 8
  %region2 = alloca [980 x i32], align 8
  %ptr1 = getelementptr inbounds [980 x i32]* %region1, i64 0, i64 1
  %ptr2 = getelementptr inbounds [980 x i32]* %region2, i64 0, i64 1
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Test the first out-of-range offset.  We cannot use an index register here.
define void @f2() {
; CHECK-NOFP: f2:
; CHECK-NOFP: lay %r1, 4096(%r15)
; CHECK-NOFP: mvhi 0(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f2:
; CHECK-FP: lay %r1, 4096(%r11)
; CHECK-FP: mvhi 0(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [980 x i32], align 8
  %region2 = alloca [980 x i32], align 8
  %ptr1 = getelementptr inbounds [980 x i32]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [980 x i32]* %region2, i64 0, i64 2
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Test the next offset after that.
define void @f3() {
; CHECK-NOFP: f3:
; CHECK-NOFP: lay %r1, 4096(%r15)
; CHECK-NOFP: mvhi 4(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f3:
; CHECK-FP: lay %r1, 4096(%r11)
; CHECK-FP: mvhi 4(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [980 x i32], align 8
  %region2 = alloca [980 x i32], align 8
  %ptr1 = getelementptr inbounds [980 x i32]* %region1, i64 0, i64 3
  %ptr2 = getelementptr inbounds [980 x i32]* %region2, i64 0, i64 3
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Add 4096 bytes (1024 words) to the size of each object and repeat.
define void @f4() {
; CHECK-NOFP: f4:
; CHECK-NOFP: lay %r1, 4096(%r15)
; CHECK-NOFP: mvhi 4092(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f4:
; CHECK-FP: lay %r1, 4096(%r11)
; CHECK-FP: mvhi 4092(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2004 x i32], align 8
  %region2 = alloca [2004 x i32], align 8
  %ptr1 = getelementptr inbounds [2004 x i32]* %region1, i64 0, i64 1
  %ptr2 = getelementptr inbounds [2004 x i32]* %region2, i64 0, i64 1
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; ...as above.
define void @f5() {
; CHECK-NOFP: f5:
; CHECK-NOFP: lay %r1, 8192(%r15)
; CHECK-NOFP: mvhi 0(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f5:
; CHECK-FP: lay %r1, 8192(%r11)
; CHECK-FP: mvhi 0(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2004 x i32], align 8
  %region2 = alloca [2004 x i32], align 8
  %ptr1 = getelementptr inbounds [2004 x i32]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [2004 x i32]* %region2, i64 0, i64 2
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; ...as above.
define void @f6() {
; CHECK-NOFP: f6:
; CHECK-NOFP: lay %r1, 8192(%r15)
; CHECK-NOFP: mvhi 4(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f6:
; CHECK-FP: lay %r1, 8192(%r11)
; CHECK-FP: mvhi 4(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2004 x i32], align 8
  %region2 = alloca [2004 x i32], align 8
  %ptr1 = getelementptr inbounds [2004 x i32]* %region1, i64 0, i64 3
  %ptr2 = getelementptr inbounds [2004 x i32]* %region2, i64 0, i64 3
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Now try an offset of 4092 from the start of the object, with the object
; being at offset 8192.  This time we need objects of (8192 - 168) / 4 = 2006
; words.
define void @f7() {
; CHECK-NOFP: f7:
; CHECK-NOFP: lay %r1, 8192(%r15)
; CHECK-NOFP: mvhi 4092(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f7:
; CHECK-FP: lay %r1, 8192(%r11)
; CHECK-FP: mvhi 4092(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2006 x i32], align 8
  %region2 = alloca [2006 x i32], align 8
  %ptr1 = getelementptr inbounds [2006 x i32]* %region1, i64 0, i64 1023
  %ptr2 = getelementptr inbounds [2006 x i32]* %region2, i64 0, i64 1023
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Keep the object-relative offset the same but bump the size of the
; objects by one doubleword.
define void @f8() {
; CHECK-NOFP: f8:
; CHECK-NOFP: lay %r1, 12288(%r15)
; CHECK-NOFP: mvhi 4(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f8:
; CHECK-FP: lay %r1, 12288(%r11)
; CHECK-FP: mvhi 4(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2008 x i32], align 8
  %region2 = alloca [2008 x i32], align 8
  %ptr1 = getelementptr inbounds [2008 x i32]* %region1, i64 0, i64 1023
  %ptr2 = getelementptr inbounds [2008 x i32]* %region2, i64 0, i64 1023
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Check a case where the original displacement is out of range.  The backend
; should force an LAY from the outset.  We don't yet do any kind of anchor
; optimization, so there should be no offset on the MVHI itself.
define void @f9() {
; CHECK-NOFP: f9:
; CHECK-NOFP: lay %r1, 12296(%r15)
; CHECK-NOFP: mvhi 0(%r1), 42
; CHECK-NOFP: br %r14
;
; CHECK-FP: f9:
; CHECK-FP: lay %r1, 12296(%r11)
; CHECK-FP: mvhi 0(%r1), 42
; CHECK-FP: br %r14
  %region1 = alloca [2008 x i32], align 8
  %region2 = alloca [2008 x i32], align 8
  %ptr1 = getelementptr inbounds [2008 x i32]* %region1, i64 0, i64 1024
  %ptr2 = getelementptr inbounds [2008 x i32]* %region2, i64 0, i64 1024
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  ret void
}

; Repeat f2 in a case that needs the emergency spill slot (because all
; call-clobbered registers are live and no call-saved ones have been
; allocated).
define void @f10(i32 *%vptr) {
; CHECK-NOFP: f10:
; CHECK-NOFP: stg [[REGISTER:%r[1-9][0-4]?]], 160(%r15)
; CHECK-NOFP: lay [[REGISTER]], 4096(%r15)
; CHECK-NOFP: mvhi 0([[REGISTER]]), 42
; CHECK-NOFP: lg [[REGISTER]], 160(%r15)
; CHECK-NOFP: br %r14
;
; CHECK-FP: f10:
; CHECK-FP: stg [[REGISTER:%r[1-9][0-4]?]], 160(%r11)
; CHECK-FP: lay [[REGISTER]], 4096(%r11)
; CHECK-FP: mvhi 0([[REGISTER]]), 42
; CHECK-FP: lg [[REGISTER]], 160(%r11)
; CHECK-FP: br %r14
  %i0 = load volatile i32 *%vptr
  %i1 = load volatile i32 *%vptr
  %i3 = load volatile i32 *%vptr
  %i4 = load volatile i32 *%vptr
  %i5 = load volatile i32 *%vptr
  %region1 = alloca [980 x i32], align 8
  %region2 = alloca [980 x i32], align 8
  %ptr1 = getelementptr inbounds [980 x i32]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [980 x i32]* %region2, i64 0, i64 2
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  store volatile i32 %i0, i32 *%vptr
  store volatile i32 %i1, i32 *%vptr
  store volatile i32 %i3, i32 *%vptr
  store volatile i32 %i4, i32 *%vptr
  store volatile i32 %i5, i32 *%vptr
  ret void
}

; And again with maximum register pressure.  The only spill slot that the
; NOFP case needs is the emergency one, so the offsets are the same as for f2.
; However, the FP case uses %r11 as the frame pointer and must therefore
; spill a second register.  This leads to an extra displacement of 8.
define void @f11(i32 *%vptr) {
; CHECK-NOFP: f11:
; CHECK-NOFP: stmg %r6, %r15,
; CHECK-NOFP: stg [[REGISTER:%r[1-9][0-4]?]], 160(%r15)
; CHECK-NOFP: lay [[REGISTER]], 4096(%r15)
; CHECK-NOFP: mvhi 0([[REGISTER]]), 42
; CHECK-NOFP: lg [[REGISTER]], 160(%r15)
; CHECK-NOFP: lmg %r6, %r15,
; CHECK-NOFP: br %r14
;
; CHECK-FP: f11:
; CHECK-FP: stmg %r6, %r15,
; CHECK-FP: stg [[REGISTER:%r[1-9][0-4]?]], 160(%r11)
; CHECK-FP: lay [[REGISTER]], 4096(%r11)
; CHECK-FP: mvhi 8([[REGISTER]]), 42
; CHECK-FP: lg [[REGISTER]], 160(%r11)
; CHECK-FP: lmg %r6, %r15,
; CHECK-FP: br %r14
  %i0 = load volatile i32 *%vptr
  %i1 = load volatile i32 *%vptr
  %i3 = load volatile i32 *%vptr
  %i4 = load volatile i32 *%vptr
  %i5 = load volatile i32 *%vptr
  %i6 = load volatile i32 *%vptr
  %i7 = load volatile i32 *%vptr
  %i8 = load volatile i32 *%vptr
  %i9 = load volatile i32 *%vptr
  %i10 = load volatile i32 *%vptr
  %i11 = load volatile i32 *%vptr
  %i12 = load volatile i32 *%vptr
  %i13 = load volatile i32 *%vptr
  %i14 = load volatile i32 *%vptr
  %region1 = alloca [980 x i32], align 8
  %region2 = alloca [980 x i32], align 8
  %ptr1 = getelementptr inbounds [980 x i32]* %region1, i64 0, i64 2
  %ptr2 = getelementptr inbounds [980 x i32]* %region2, i64 0, i64 2
  store volatile i32 42, i32 *%ptr1
  store volatile i32 42, i32 *%ptr2
  store volatile i32 %i0, i32 *%vptr
  store volatile i32 %i1, i32 *%vptr
  store volatile i32 %i3, i32 *%vptr
  store volatile i32 %i4, i32 *%vptr
  store volatile i32 %i5, i32 *%vptr
  store volatile i32 %i6, i32 *%vptr
  store volatile i32 %i7, i32 *%vptr
  store volatile i32 %i8, i32 *%vptr
  store volatile i32 %i9, i32 *%vptr
  store volatile i32 %i10, i32 *%vptr
  store volatile i32 %i11, i32 *%vptr
  store volatile i32 %i12, i32 *%vptr
  store volatile i32 %i13, i32 *%vptr
  store volatile i32 %i14, i32 *%vptr
  ret void
}

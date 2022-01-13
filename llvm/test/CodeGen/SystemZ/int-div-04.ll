; Testg 64-bit signed division and remainder.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -asm-verbose=0 | FileCheck %s

declare i64 @foo()

; Testg register division.  The result is in the second of the two registers.
define void @f1(i64 %dummy, i64 %a, i64 %b, i64 *%dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgr %r2, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %div = sdiv i64 %a, %b
  store i64 %div, i64 *%dest
  ret void
}

; Testg register remainder.  The result is in the first of the two registers.
define void @f2(i64 %dummy, i64 %a, i64 %b, i64 *%dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgr %r2, %r4
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %rem = srem i64 %a, %b
  store i64 %rem, i64 *%dest
  ret void
}

; Testg that division and remainder use a single instruction.
define i64 @f3(i64 %dummy1, i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgr %r2, %r4
; CHECK-NOT: dsgr
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Testg memory division with no displacement.
define void @f4(i64 %dummy, i64 %a, i64 *%src, i64 *%dest) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsg %r2, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %div = sdiv i64 %a, %b
  store i64 %div, i64 *%dest
  ret void
}

; Testg memory remainder with no displacement.
define void @f5(i64 %dummy, i64 %a, i64 *%src, i64 *%dest) {
; CHECK-LABEL: f5:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsg %r2, 0(%r4)
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %rem = srem i64 %a, %b
  store i64 %rem, i64 *%dest
  ret void
}

; Testg both memory division and memory remainder.
define i64 @f6(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsg %r2, 0(%r4)
; CHECK-NOT: {{dsg|dsgr}}
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %b = load i64, i64 *%src
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Check the high end of the DSG range.
define i64 @f7(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: dsg %r2, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, 524288
; CHECK: dsg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check the high end of the negative aligned DSG range.
define i64 @f9(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f9:
; CHECK: dsg %r2, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check the low end of the DSG range.
define i64 @f10(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f10:
; CHECK: dsg %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f11(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r4, -524296
; CHECK: dsg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check that DSG allows an index.
define i64 @f12(i64 %dummy, i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: dsg %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %rem = srem i64 %a, %b
  ret i64 %rem
}

; Check that divisions of spilled values can use DSG rather than DSGR.
define i64 @f13(i64 *%ptr0) {
; CHECK-LABEL: f13:
; CHECK: brasl %r14, foo@PLT
; CHECK: dsg {{%r[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16
  %ptr9 = getelementptr i64, i64 *%ptr0, i64 18
  %ptr10 = getelementptr i64, i64 *%ptr0, i64 20

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8
  %val9 = load i64, i64 *%ptr9
  %val10 = load i64, i64 *%ptr10

  %ret = call i64 @foo()

  %div0 = sdiv i64 %ret, %val0
  %div1 = sdiv i64 %div0, %val1
  %div2 = sdiv i64 %div1, %val2
  %div3 = sdiv i64 %div2, %val3
  %div4 = sdiv i64 %div3, %val4
  %div5 = sdiv i64 %div4, %val5
  %div6 = sdiv i64 %div5, %val6
  %div7 = sdiv i64 %div6, %val7
  %div8 = sdiv i64 %div7, %val8
  %div9 = sdiv i64 %div8, %val9
  %div10 = sdiv i64 %div9, %val10

  ret i64 %div10
}

; RUN: llc -O0 -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; Test case for PR 14779: anonymous aggregates are not handled correctly.
; The bug is triggered by passing a byval structure after an anonymous
; aggregate.

%tarray = type { i64, i8* }

define i8* @func1({ i64, i8* } %array, i8* %ptr) {
entry:
  %array_ptr = extractvalue {i64, i8* } %array, 1
  %cond = icmp eq i8* %array_ptr, %ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array_ptr
unequal:
  ret i8* %ptr
}

; CHECK: func1:
; CHECK: cmpld {{[0-9]+}}, 4, 5
; CHECK: std 4, -[[OFFSET1:[0-9]+]]
; CHECK: std 5, -[[OFFSET2:[0-9]+]]
; CHECK: ld 3, -[[OFFSET1]](1)
; CHECK: ld 3, -[[OFFSET2]](1)


define i8* @func2({ i64, i8* } %array1, %tarray* byval %array2) {
entry:
  %array1_ptr = extractvalue {i64, i8* } %array1, 1
  %tmp = getelementptr inbounds %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8** %tmp
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}

; CHECK: func2:
; CHECK: addi [[REG1:[0-9]+]], 1, 64
; CHECK: ld [[REG2:[0-9]+]], 8([[REG1]])
; CHECK: cmpld {{[0-9]+}}, 4, [[REG2]]
; CHECK: std [[REG2]], -[[OFFSET1:[0-9]+]]
; CHECK: std 4, -[[OFFSET2:[0-9]+]]
; CHECK: ld 3, -[[OFFSET2]](1)
; CHECK: ld 3, -[[OFFSET1]](1)

define i8* @func3({ i64, i8* }* byval %array1, %tarray* byval %array2) {
entry:
  %tmp1 = getelementptr inbounds { i64, i8* }* %array1, i32 0, i32 1
  %array1_ptr = load i8** %tmp1
  %tmp2 = getelementptr inbounds %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8** %tmp2
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}

; CHECK: func3:
; CHECK: addi [[REG1:[0-9]+]], 1, 64
; CHECK: addi [[REG2:[0-9]+]], 1, 48
; CHECK: ld [[REG3:[0-9]+]], 8([[REG1]])
; CHECK: ld [[REG4:[0-9]+]], 8([[REG2]])
; CHECK: cmpld {{[0-9]+}}, [[REG4]], [[REG3]]
; CHECK: std [[REG3]], -[[OFFSET1:[0-9]+]](1)
; CHECK: std [[REG4]], -[[OFFSET2:[0-9]+]](1)
; CHECK: ld 3, -[[OFFSET2]](1)
; CHECK: ld 3, -[[OFFSET1]](1)

define i8* @func4(i64 %p1, i64 %p2, i64 %p3, i64 %p4,
                  i64 %p5, i64 %p6, i64 %p7, i64 %p8,
                  { i64, i8* } %array1, %tarray* byval %array2) {
entry:
  %array1_ptr = extractvalue {i64, i8* } %array1, 1
  %tmp = getelementptr inbounds %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8** %tmp
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}

; CHECK: func4:
; CHECK: addi [[REG1:[0-9]+]], 1, 128
; CHECK: ld [[REG2:[0-9]+]], 120(1)
; CHECK: ld [[REG3:[0-9]+]], 8([[REG1]])
; CHECK: cmpld {{[0-9]+}}, [[REG2]], [[REG3]]
; CHECK: std [[REG2]], -[[OFFSET1:[0-9]+]](1)
; CHECK: std [[REG3]], -[[OFFSET2:[0-9]+]](1)
; CHECK: ld 3, -[[OFFSET1]](1)
; CHECK: ld 3, -[[OFFSET2]](1)


; Test high-word operations, using "h" constraints to force a high
; register and "r" constraints to force a low register.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test loads and stores involving mixtures of high and low registers.
define void @f1(i32 *%ptr1, i32 *%ptr2) {
; CHECK-LABEL: f1:
; CHECK-DAG: lfh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: l [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lfh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: ly [[REG4:%r[0-5]]], 524284(%r3)
; CHECK: blah [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: stfh [[REG1]], 0(%r2)
; CHECK-DAG: st [[REG2]], 0(%r3)
; CHECK-DAG: stfh [[REG3]], 4096(%r2)
; CHECK-DAG: sty [[REG4]], 524284(%r3)
; CHECK: br %r14
  %ptr3 = getelementptr i32 *%ptr1, i64 1024
  %ptr4 = getelementptr i32 *%ptr2, i64 131071
  %old1 = load i32 *%ptr1
  %old2 = load i32 *%ptr2
  %old3 = load i32 *%ptr3
  %old4 = load i32 *%ptr4
  %res = call { i32, i32, i32, i32 } asm "blah $0, $1, $2, $3",
              "=h,=r,=h,=r,0,1,2,3"(i32 %old1, i32 %old2, i32 %old3, i32 %old4)
  %new1 = extractvalue { i32, i32, i32, i32 } %res, 0
  %new2 = extractvalue { i32, i32, i32, i32 } %res, 1
  %new3 = extractvalue { i32, i32, i32, i32 } %res, 2
  %new4 = extractvalue { i32, i32, i32, i32 } %res, 3
  store i32 %new1, i32 *%ptr1
  store i32 %new2, i32 *%ptr2
  store i32 %new3, i32 *%ptr3
  store i32 %new4, i32 *%ptr4
  ret void
}

; Test moves involving mixtures of high and low registers.
define i32 @f2(i32 %old) {
; CHECK-LABEL: f2:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 0, 159, 32
; CHECK-DAG: lr %r3, %r2
; CHECK: stepa [[REG1]], %r2, %r3
; CHECK: risbhg {{%r[0-5]}}, [[REG1]], 0, 159, 0
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: risblg %r2, [[REG2]], 0, 159, 32
; CHECK: br %r14
  %tmp = call i32 asm "stepa $1, $2, $3",
                      "=h,0,{r2},{r3}"(i32 %old, i32 %old, i32 %old)
  %new = call i32 asm "stepb $1, $2", "=&h,0,h"(i32 %tmp, i32 %tmp)
  ret i32 %new
}

; Test sign-extending 8-bit loads into mixtures of high and low registers.
define void @f3(i8 *%ptr1, i8 *%ptr2) {
; CHECK-LABEL: f3:
; CHECK-DAG: lbh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: lb [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lbh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: lb [[REG4:%r[0-5]]], 524287(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i8 *%ptr1, i64 4096
  %ptr4 = getelementptr i8 *%ptr2, i64 524287
  %val1 = load i8 *%ptr1
  %val2 = load i8 *%ptr2
  %val3 = load i8 *%ptr3
  %val4 = load i8 *%ptr4
  %ext1 = sext i8 %val1 to i32
  %ext2 = sext i8 %val2 to i32
  %ext3 = sext i8 %val3 to i32
  %ext4 = sext i8 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test sign-extending 16-bit loads into mixtures of high and low registers.
define void @f4(i16 *%ptr1, i16 *%ptr2) {
; CHECK-LABEL: f4:
; CHECK-DAG: lhh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: lh [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lhh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: lhy [[REG4:%r[0-5]]], 524286(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i16 *%ptr1, i64 2048
  %ptr4 = getelementptr i16 *%ptr2, i64 262143
  %val1 = load i16 *%ptr1
  %val2 = load i16 *%ptr2
  %val3 = load i16 *%ptr3
  %val4 = load i16 *%ptr4
  %ext1 = sext i16 %val1 to i32
  %ext2 = sext i16 %val2 to i32
  %ext3 = sext i16 %val3 to i32
  %ext4 = sext i16 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

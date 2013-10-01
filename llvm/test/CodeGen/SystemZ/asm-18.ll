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

; Test zero-extending 8-bit loads into mixtures of high and low registers.
define void @f5(i8 *%ptr1, i8 *%ptr2) {
; CHECK-LABEL: f5:
; CHECK-DAG: llch [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: llc [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: llch [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: llc [[REG4:%r[0-5]]], 524287(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i8 *%ptr1, i64 4096
  %ptr4 = getelementptr i8 *%ptr2, i64 524287
  %val1 = load i8 *%ptr1
  %val2 = load i8 *%ptr2
  %val3 = load i8 *%ptr3
  %val4 = load i8 *%ptr4
  %ext1 = zext i8 %val1 to i32
  %ext2 = zext i8 %val2 to i32
  %ext3 = zext i8 %val3 to i32
  %ext4 = zext i8 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test zero-extending 16-bit loads into mixtures of high and low registers.
define void @f6(i16 *%ptr1, i16 *%ptr2) {
; CHECK-LABEL: f6:
; CHECK-DAG: llhh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: llh [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: llhh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: llh [[REG4:%r[0-5]]], 524286(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i16 *%ptr1, i64 2048
  %ptr4 = getelementptr i16 *%ptr2, i64 262143
  %val1 = load i16 *%ptr1
  %val2 = load i16 *%ptr2
  %val3 = load i16 *%ptr3
  %val4 = load i16 *%ptr4
  %ext1 = zext i16 %val1 to i32
  %ext2 = zext i16 %val2 to i32
  %ext3 = zext i16 %val3 to i32
  %ext4 = zext i16 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test truncating stores of high and low registers into 8-bit memory.
define void @f7(i8 *%ptr1, i8 *%ptr2) {
; CHECK-LABEL: f7:
; CHECK: blah [[REG1:%r[0-5]]], [[REG2:%r[0-5]]]
; CHECK-DAG: stch [[REG1]], 0(%r2)
; CHECK-DAG: stc [[REG2]], 0(%r3)
; CHECK-DAG: stch [[REG1]], 4096(%r2)
; CHECK-DAG: stcy [[REG2]], 524287(%r3)
; CHECK: br %r14
  %res = call { i32, i32 } asm "blah $0, $1", "=h,=r"()
  %res1 = extractvalue { i32, i32 } %res, 0
  %res2 = extractvalue { i32, i32 } %res, 1
  %trunc1 = trunc i32 %res1 to i8
  %trunc2 = trunc i32 %res2 to i8
  %ptr3 = getelementptr i8 *%ptr1, i64 4096
  %ptr4 = getelementptr i8 *%ptr2, i64 524287
  store i8 %trunc1, i8 *%ptr1
  store i8 %trunc2, i8 *%ptr2
  store i8 %trunc1, i8 *%ptr3
  store i8 %trunc2, i8 *%ptr4
  ret void
}

; Test truncating stores of high and low registers into 16-bit memory.
define void @f8(i16 *%ptr1, i16 *%ptr2) {
; CHECK-LABEL: f8:
; CHECK: blah [[REG1:%r[0-5]]], [[REG2:%r[0-5]]]
; CHECK-DAG: sthh [[REG1]], 0(%r2)
; CHECK-DAG: sth [[REG2]], 0(%r3)
; CHECK-DAG: sthh [[REG1]], 4096(%r2)
; CHECK-DAG: sthy [[REG2]], 524286(%r3)
; CHECK: br %r14
  %res = call { i32, i32 } asm "blah $0, $1", "=h,=r"()
  %res1 = extractvalue { i32, i32 } %res, 0
  %res2 = extractvalue { i32, i32 } %res, 1
  %trunc1 = trunc i32 %res1 to i16
  %trunc2 = trunc i32 %res2 to i16
  %ptr3 = getelementptr i16 *%ptr1, i64 2048
  %ptr4 = getelementptr i16 *%ptr2, i64 262143
  store i16 %trunc1, i16 *%ptr1
  store i16 %trunc2, i16 *%ptr2
  store i16 %trunc1, i16 *%ptr3
  store i16 %trunc2, i16 *%ptr4
  ret void
}

; Test zero extensions from 8 bits between mixtures of high and low registers.
define i32 @f9(i8 %val1, i8 %val2) {
; CHECK-LABEL: f9:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 24, 159, 32
; CHECK-DAG: llcr [[REG2:%r[0-5]]], %r3
; CHECK: stepa [[REG1]], [[REG2]]
; CHECK: risbhg [[REG3:%r[0-5]]], [[REG1]], 24, 159, 0
; CHECK: stepb [[REG3]]
; CHECK: risblg %r2, [[REG3]], 24, 159, 32
; CHECK: br %r14
  %ext1 = zext i8 %val1 to i32
  %ext2 = zext i8 %val2 to i32
  %val3 = call i8 asm sideeffect "stepa $0, $1", "=h,0,r"(i32 %ext1, i32 %ext2)
  %ext3 = zext i8 %val3 to i32
  %val4 = call i8 asm sideeffect "stepb $0", "=h,0"(i32 %ext3)
  %ext4 = zext i8 %val4 to i32
  ret i32 %ext4
}

; Test zero extensions from 16 bits between mixtures of high and low registers.
define i32 @f10(i16 %val1, i16 %val2) {
; CHECK-LABEL: f10:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 16, 159, 32
; CHECK-DAG: llhr [[REG2:%r[0-5]]], %r3
; CHECK: stepa [[REG1]], [[REG2]]
; CHECK: risbhg [[REG3:%r[0-5]]], [[REG1]], 16, 159, 0
; CHECK: stepb [[REG3]]
; CHECK: risblg %r2, [[REG3]], 16, 159, 32
; CHECK: br %r14
  %ext1 = zext i16 %val1 to i32
  %ext2 = zext i16 %val2 to i32
  %val3 = call i16 asm sideeffect "stepa $0, $1", "=h,0,r"(i32 %ext1, i32 %ext2)
  %ext3 = zext i16 %val3 to i32
  %val4 = call i16 asm sideeffect "stepb $0", "=h,0"(i32 %ext3)
  %ext4 = zext i16 %val4 to i32
  ret i32 %ext4
}

; Test loads of 16-bit constants into mixtures of high and low registers.
define void @f11() {
; CHECK-LABEL: f11:
; CHECK-DAG: iihf [[REG1:%r[0-5]]], 4294934529
; CHECK-DAG: lhi [[REG2:%r[0-5]]], -32768
; CHECK-DAG: llihl [[REG3:%r[0-5]]], 32766
; CHECK-DAG: lhi [[REG4:%r[0-5]]], 32767
; CHECK: blah [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK: br %r14
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 -32767, i32 -32768,
                                     i32 32766, i32 32767)
  ret void
}

; Test loads of unsigned constants into mixtures of high and low registers.
; For stepc, we expect the h and r operands to be paired by the register
; allocator.  It doesn't really matter which comes first: LLILL/IIHF would
; be just as good.
define void @f12() {
; CHECK-LABEL: f12:
; CHECK-DAG: llihl [[REG1:%r[0-5]]], 32768
; CHECK-DAG: llihl [[REG2:%r[0-5]]], 65535
; CHECK-DAG: llihh [[REG3:%r[0-5]]], 1
; CHECK-DAG: llihh [[REG4:%r[0-5]]], 65535
; CHECK: stepa [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: llill [[REG1:%r[0-5]]], 32769
; CHECK-DAG: llill [[REG2:%r[0-5]]], 65534
; CHECK-DAG: llilh [[REG3:%r[0-5]]], 2
; CHECK-DAG: llilh [[REG4:%r[0-5]]], 65534
; CHECK: stepb [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: llihl [[REG1:%r[0-5]]], 32770
; CHECK-DAG: iilf [[REG1]], 65533
; CHECK-DAG: llihh [[REG2:%r[0-5]]], 4
; CHECK-DAG: iilf [[REG2]], 524288
; CHECK: stepc [[REG1]], [[REG1]], [[REG2]], [[REG2]]
; CHECK-DAG: iihf [[REG1:%r[0-5]]], 3294967296
; CHECK-DAG: iilf [[REG2:%r[0-5]]], 4294567296
; CHECK-DAG: iihf [[REG3:%r[0-5]]], 1000000000
; CHECK-DAG: iilf [[REG4:%r[0-5]]], 400000
; CHECK: stepd [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK: br %r14
  call void asm sideeffect "stepa $0, $1, $2, $3",
                           "h,h,h,h"(i32 32768, i32 65535,
                                     i32 65536, i32 -65536)
  call void asm sideeffect "stepb $0, $1, $2, $3",
                           "r,r,r,r"(i32 32769, i32 65534,
                                     i32 131072, i32 -131072)
  call void asm sideeffect "stepc $0, $1, $2, $3",
                           "h,r,h,r"(i32 32770, i32 65533,
                                     i32 262144, i32 524288)
  call void asm sideeffect "stepd $0, $1, $2, $3",
                           "h,r,h,r"(i32 -1000000000, i32 -400000,
                                     i32 1000000000, i32 400000)
  ret void
}

; Test selects involving high registers.
define void @f13(i32 %x, i32 %y) {
; CHECK-LABEL: f13:
; CHECK: llihl [[REG:%r[0-5]]], 0
; CHECK: cije %r2, 0
; CHECK: iihf [[REG]], 2102030405
; CHECK: blah [[REG]]
; CHECK: br %r14
  %cmp = icmp eq i32 %x, 0
  %val = select i1 %cmp, i32 0, i32 2102030405
  call void asm sideeffect "blah $0", "h"(i32 %val)
  ret void
}

; Test selects involving low registers.
define void @f14(i32 %x, i32 %y) {
; CHECK-LABEL: f14:
; CHECK: lhi [[REG:%r[0-5]]], 0
; CHECK: cije %r2, 0
; CHECK: iilf [[REG]], 2102030405
; CHECK: blah [[REG]]
; CHECK: br %r14
  %cmp = icmp eq i32 %x, 0
  %val = select i1 %cmp, i32 0, i32 2102030405
  call void asm sideeffect "blah $0", "r"(i32 %val)
  ret void
}

; Test immediate insertion involving high registers.
define void @f15() {
; CHECK-LABEL: f15:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: iihh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: iihl [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %and1 = and i32 %res1, 65535
  %or1 = or i32 %and1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %or1)
  %and2 = and i32 %res2, -65536
  %or2 = or i32 %and2, 34661
  call void asm sideeffect "stepc $0", "h"(i32 %or2)
  ret void
}

; Test immediate insertion involving low registers.
define void @f16() {
; CHECK-LABEL: f16:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: iilh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: iill [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %and1 = and i32 %res1, 65535
  %or1 = or i32 %and1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %or1)
  %and2 = and i32 %res2, -65536
  %or2 = or i32 %and2, 34661
  call void asm sideeffect "stepc $0", "r"(i32 %or2)
  ret void
}

; Test immediate OR involving high registers.
define void @f17() {
; CHECK-LABEL: f17:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: oihh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: oihl [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: oihf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %or1 = or i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %or1)
  %or2 = or i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %or2)
  %or3 = or i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "h"(i32 %or3)
  ret void
}

; Test immediate OR involving low registers.
define void @f18() {
; CHECK-LABEL: f18:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: oilh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: oill [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: oilf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %or1 = or i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %or1)
  %or2 = or i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %or2)
  %or3 = or i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "r"(i32 %or3)
  ret void
}

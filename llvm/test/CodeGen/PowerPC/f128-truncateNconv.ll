; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs -enable-ppc-quad-precision \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s

@f128Array = global [4 x fp128] [fp128 0xL00000000000000004004C00000000000,
                                 fp128 0xLF000000000000000400808AB851EB851,
                                 fp128 0xL5000000000000000400E0C26324C8366,
                                 fp128 0xL8000000000000000400A24E2E147AE14],
                                align 16

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2sdw(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2sdw
; CHECK: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xscvqpsdz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: mfvsrd 3, [[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_02(i64* nocapture %res) local_unnamed_addr #1 {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptosi fp128 %0 to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_02
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK: lxv [[REG:[0-9]+]], 32([[REG0]])
; CHECK-NEXT: xscvqpsdz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxsd [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2sdw_03(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2sdw_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK-DAG: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK-DAG: lxv [[REG1:[0-9]+]], 16([[REG0]])
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK: xsaddqp [[REG]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpsdz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: mfvsrd 3, [[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_04
; CHECK-DAG: lxv [[REG1:[0-9]+]], 0(4)
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK: xsaddqp [[REG]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpsdz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxsd [[CONV]], 0(5)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_testXForm(i64* nocapture %res, i32 signext %idx) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptosi fp128 %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %res, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_testXForm
; CHECK: xscvqpsdz [[CONV:[0-9]+]],
; CHECK-NEXT: stxsdx [[CONV]], 3, 4
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2udw(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2udw
; CHECK: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xscvqpudz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: mfvsrd 3, [[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_02(i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptoui fp128 %0 to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2udw_02
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK: lxv [[REG:[0-9]+]], 32([[REG0]])
; CHECK-NEXT: xscvqpudz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxsd [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2udw_03(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2udw_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK-DAG: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK-DAG: lxv [[REG1:[0-9]+]], 16([[REG0]])
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK: xsaddqp [[REG]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpudz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: mfvsrd 3, [[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2udw_04
; CHECK-DAG: lxv [[REG1:[0-9]+]], 0(4)
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK: xsaddqp [[REG]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpudz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxsd [[CONV]], 0(5)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_testXForm(i64* nocapture %res, i32 signext %idx) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 0), align 16
  %conv = fptoui fp128 %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %res, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: qpConv2udw_testXForm
; CHECK: xscvqpudz [[CONV:[0-9]+]],
; CHECK-NEXT: stxsdx [[CONV]], 3, 4
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @qpConv2sw(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2sw
; CHECK: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xscvqpswz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: mfvsrwz [[REG2:[0-9]+]], [[CONV]]
; CHECK-NEXT: extsw 3, [[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sw_02(i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptosi fp128 %0 to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2sw_02
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK: lxv [[REG:[0-9]+]], 32([[REG0]])
; CHECK-NEXT: xscvqpswz [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxsiwx [[CONV]], 0, 3
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @qpConv2sw_03(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2sw_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK-DAG: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK-DAG: lxv [[REG1:[0-9]+]], 16([[REG0]])
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xsaddqp [[ADD:[0-9]+]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpswz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: mfvsrwz [[REG2:[0-9]+]], [[CONV]]
; CHECK-NEXT: extsw 3, [[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sw_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2sw_04
; CHECK-DAG: lxv [[REG1:[0-9]+]], 0(4)
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xsaddqp [[ADD:[0-9]+]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpswz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: stxsiwx [[CONV]], 0, 5
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @qpConv2uw(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2uw
; CHECK: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xscvqpuwz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: mfvsrwz 3, [[CONV]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2uw_02(i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptoui fp128 %0 to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2uw_02
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK: lxv [[REG:[0-9]+]], 32([[REG0]])
; CHECK-NEXT: xscvqpuwz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: stxsiwx [[CONV]], 0, 3
; CHECK: blr
}

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @qpConv2uw_03(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2uw_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC0@toc@ha
; CHECK-DAG: ld [[REG0]], .LC0@toc@l([[REG0]])
; CHECK-DAG: lxv [[REG1:[0-9]+]], 16([[REG0]])
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xsaddqp [[ADD:[0-9]+]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpuwz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: mfvsrwz 3, [[CONV]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2uw_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2uw_04
; CHECK-DAG: lxv [[REG1:[0-9]+]], 0(4)
; CHECK-DAG: lxv [[REG:[0-9]+]], 0(3)
; CHECK-NEXT: xsaddqp [[ADD:[0-9]+]], [[REG]], [[REG1]]
; CHECK-NEXT: xscvqpuwz [[CONV:[0-9]+]], [[ADD]]
; CHECK-NEXT: stxsiwx [[CONV]], 0, 5
; CHECK: blr
}

; RUN: llc -O3 -mv67t -march=hexagon < %s | FileCheck %s

; Test that the inner loop in the tiny core version of bkfir has the assembler
; directive "p2align 4".

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK-NOT: falign
; CHECK: p2align 4
; CHECK: } :endloop0

define void @bkfir(i32* nocapture readonly %in, i32* nocapture readonly %coefs, i32 %tap, i32 %length, i32* nocapture %out) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %out to i64*
  %cmp141 = icmp sgt i32 %length, 0
  br i1 %cmp141, label %for.body.lr.ph, label %for.end52

for.body.lr.ph:
  %1 = bitcast i32* %coefs to i64*
  %cmp8127 = icmp sgt i32 %tap, 0
  br i1 %cmp8127, label %for.body.us.preheader, label %for.body.lr.ph.split

for.body.us.preheader:
  br label %for.body.us

for.body.us:
  %add.ptr.us.phi = phi i32* [ %add.ptr.us.inc, %for.cond7.for.end_crit_edge.us ], [ %in, %for.body.us.preheader ]
  %i.0143.us = phi i32 [ %add51.us, %for.cond7.for.end_crit_edge.us ], [ 0, %for.body.us.preheader ]
  %optr.0142.us = phi i64* [ %incdec.ptr49.us, %for.cond7.for.end_crit_edge.us ], [ %0, %for.body.us.preheader ]
  %2 = bitcast i32* %add.ptr.us.phi to i64*
  %incdec.ptr.us = getelementptr inbounds i32, i32* %add.ptr.us.phi, i32 2
  %3 = bitcast i32* %incdec.ptr.us to i64*
  %4 = load i64, i64* %2, align 8
  %incdec.ptr1.us = getelementptr inbounds i32, i32* %add.ptr.us.phi, i32 4
  %5 = bitcast i32* %incdec.ptr1.us to i64*
  %6 = load i64, i64* %3, align 8
  %_Q6V64_internal_union.sroa.0.0.extract.trunc.us = trunc i64 %6 to i32
  %_Q6V64_internal_union2.sroa.3.0.extract.shift.us = lshr i64 %4, 32
  %_Q6V64_internal_union2.sroa.3.0.extract.trunc.us = trunc i64 %_Q6V64_internal_union2.sroa.3.0.extract.shift.us to i32
  %7 = tail call i64 @llvm.hexagon.A2.combinew(i32 %_Q6V64_internal_union.sroa.0.0.extract.trunc.us, i32 %_Q6V64_internal_union2.sroa.3.0.extract.trunc.us)
  %add.ptr.us.inc = getelementptr i32, i32* %add.ptr.us.phi, i32 4
  br label %for.body9.us

for.body9.us:
  %j.0137.us = phi i32 [ 0, %for.body.us ], [ %add.us, %for.body9.us ]
  %x0x1.0136.us = phi i64 [ %4, %for.body.us ], [ %10, %for.body9.us ]
  %x2x3.0135.us = phi i64 [ %6, %for.body.us ], [ %11, %for.body9.us ]
  %x1x2.0134.us = phi i64 [ %7, %for.body.us ], [ %13, %for.body9.us ]
  %iptrD.0133.us = phi i64* [ %5, %for.body.us ], [ %incdec.ptr13.us, %for.body9.us ]
  %iptrC.0132.us = phi i64* [ %1, %for.body.us ], [ %incdec.ptr11.us, %for.body9.us ]
  %sum0.0131.us = phi i64 [ 0, %for.body.us ], [ %18, %for.body9.us ]
  %sum1.0130.us = phi i64 [ 0, %for.body.us ], [ %19, %for.body9.us ]
  %sum2.0129.us = phi i64 [ 0, %for.body.us ], [ %20, %for.body9.us ]
  %sum3.0128.us = phi i64 [ 0, %for.body.us ], [ %21, %for.body9.us ]
  %incdec.ptr10.us = getelementptr inbounds i64, i64* %iptrC.0132.us, i32 1
  %8 = load i64, i64* %iptrC.0132.us, align 8
  %incdec.ptr11.us = getelementptr inbounds i64, i64* %iptrC.0132.us, i32 2
  %9 = load i64, i64* %incdec.ptr10.us, align 8
  %incdec.ptr12.us = getelementptr inbounds i64, i64* %iptrD.0133.us, i32 1
  %10 = load i64, i64* %iptrD.0133.us, align 8
  %incdec.ptr13.us = getelementptr inbounds i64, i64* %iptrD.0133.us, i32 2
  %11 = load i64, i64* %incdec.ptr12.us, align 8
  %_Q6V64_internal_union14.sroa.0.0.extract.trunc.us = trunc i64 %10 to i32
  %_Q6V64_internal_union14.sroa.4.0.extract.shift.us = lshr i64 %10, 32
  %_Q6V64_internal_union19.sroa.3.0.extract.shift.us = lshr i64 %x2x3.0135.us, 32
  %_Q6V64_internal_union19.sroa.3.0.extract.trunc.us = trunc i64 %_Q6V64_internal_union19.sroa.3.0.extract.shift.us to i32
  %12 = tail call i64 @llvm.hexagon.A2.combinew(i32 %_Q6V64_internal_union14.sroa.0.0.extract.trunc.us, i32 %_Q6V64_internal_union19.sroa.3.0.extract.trunc.us)
  %_Q6V64_internal_union24.sroa.0.0.extract.trunc.us = trunc i64 %11 to i32
  %_Q6V64_internal_union29.sroa.3.0.extract.trunc.us = trunc i64 %_Q6V64_internal_union14.sroa.4.0.extract.shift.us to i32
  %13 = tail call i64 @llvm.hexagon.A2.combinew(i32 %_Q6V64_internal_union24.sroa.0.0.extract.trunc.us, i32 %_Q6V64_internal_union29.sroa.3.0.extract.trunc.us)
  %14 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %sum0.0131.us, i64 %x0x1.0136.us, i64 %8)
  %15 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %sum1.0130.us, i64 %x1x2.0134.us, i64 %8)
  %16 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %sum2.0129.us, i64 %x2x3.0135.us, i64 %8)
  %17 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %sum3.0128.us, i64 %12, i64 %8)
  %18 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %14, i64 %x2x3.0135.us, i64 %9)
  %19 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %15, i64 %12, i64 %9)
  %20 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %16, i64 %10, i64 %9)
  %21 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %17, i64 %13, i64 %9)
  %add.us = add nuw nsw i32 %j.0137.us, 4
  %cmp8.us = icmp slt i32 %add.us, %tap
  br i1 %cmp8.us, label %for.body9.us, label %for.cond7.for.end_crit_edge.us

for.cond7.for.end_crit_edge.us:
  %22 = ashr i64 %18, 39
  %23 = ashr i64 %19, 39
  %24 = ashr i64 %20, 39
  %25 = ashr i64 %21, 39
  %26 = tail call i32 @llvm.hexagon.A2.sat(i64 %22)
  %27 = tail call i32 @llvm.hexagon.A2.sat(i64 %23)
  %28 = tail call i32 @llvm.hexagon.A2.sat(i64 %24)
  %29 = tail call i32 @llvm.hexagon.A2.sat(i64 %25)
  %_Q6V64_internal_union34.sroa.4.0.insert.ext.us = zext i32 %27 to i64
  %_Q6V64_internal_union34.sroa.4.0.insert.shift.us = shl nuw i64 %_Q6V64_internal_union34.sroa.4.0.insert.ext.us, 32
  %_Q6V64_internal_union34.sroa.0.0.insert.ext.us = zext i32 %26 to i64
  %_Q6V64_internal_union34.sroa.0.0.insert.insert.us = or i64 %_Q6V64_internal_union34.sroa.4.0.insert.shift.us, %_Q6V64_internal_union34.sroa.0.0.insert.ext.us
  %incdec.ptr41.us = getelementptr inbounds i64, i64* %optr.0142.us, i32 1
  store i64 %_Q6V64_internal_union34.sroa.0.0.insert.insert.us, i64* %optr.0142.us, align 8
  %_Q6V64_internal_union42.sroa.4.0.insert.ext.us = zext i32 %29 to i64
  %_Q6V64_internal_union42.sroa.4.0.insert.shift.us = shl nuw i64 %_Q6V64_internal_union42.sroa.4.0.insert.ext.us, 32
  %_Q6V64_internal_union42.sroa.0.0.insert.ext.us = zext i32 %28 to i64
  %_Q6V64_internal_union42.sroa.0.0.insert.insert.us = or i64 %_Q6V64_internal_union42.sroa.4.0.insert.shift.us, %_Q6V64_internal_union42.sroa.0.0.insert.ext.us
  %incdec.ptr49.us = getelementptr inbounds i64, i64* %optr.0142.us, i32 2
  store i64 %_Q6V64_internal_union42.sroa.0.0.insert.insert.us, i64* %incdec.ptr41.us, align 8
  %add51.us = add nuw nsw i32 %i.0143.us, 4
  %cmp.us = icmp slt i32 %add51.us, %length
  br i1 %cmp.us, label %for.body.us, label %for.end52

for.body.lr.ph.split:
  %30 = tail call i32 @llvm.hexagon.A2.sat(i64 0)
  %_Q6V64_internal_union34.sroa.4.0.insert.ext = zext i32 %30 to i64
  %_Q6V64_internal_union34.sroa.4.0.insert.shift = shl nuw i64 %_Q6V64_internal_union34.sroa.4.0.insert.ext, 32
  %_Q6V64_internal_union34.sroa.0.0.insert.insert = or i64 %_Q6V64_internal_union34.sroa.4.0.insert.shift, %_Q6V64_internal_union34.sroa.4.0.insert.ext
  br label %for.body

for.body:
  %i.0143 = phi i32 [ 0, %for.body.lr.ph.split ], [ %add51, %for.body ]
  %optr.0142 = phi i64* [ %0, %for.body.lr.ph.split ], [ %incdec.ptr49, %for.body ]
  %incdec.ptr41 = getelementptr inbounds i64, i64* %optr.0142, i32 1
  store i64 %_Q6V64_internal_union34.sroa.0.0.insert.insert, i64* %optr.0142, align 8
  %incdec.ptr49 = getelementptr inbounds i64, i64* %optr.0142, i32 2
  store i64 %_Q6V64_internal_union34.sroa.0.0.insert.insert, i64* %incdec.ptr41, align 8
  %add51 = add nuw nsw i32 %i.0143, 4
  %cmp = icmp slt i32 %add51, %length
  br i1 %cmp, label %for.body, label %for.end52

for.end52:
  ret void
}

declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1
declare i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64, i64, i64) #1
declare i32 @llvm.hexagon.A2.sat(i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv67t" "target-features"="+audio" }
attributes #1 = { nounwind readnone }

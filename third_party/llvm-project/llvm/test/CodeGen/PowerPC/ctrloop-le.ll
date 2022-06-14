; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; CHECK: test_pos1_ir_sle
; CHECK: bdnz
; a < b
define void @test_pos1_ir_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 28395, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 28395, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_ir_sle
; CHECK: bdnz
; a < b
define void @test_pos2_ir_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 9073, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 9073, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_ir_sle
; CHECK: bdnz
; a < b
define void @test_pos4_ir_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 21956, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 21956, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_ir_sle
; CHECK: bdnz
; a < b
define void @test_pos8_ir_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 16782, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 16782, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_ir_sle
; CHECK: bdnz
; a < b
define void @test_pos16_ir_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 19097, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 19097, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos1_ri_sle
; CHECK: bdnz
; a < b
define void @test_pos1_ri_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, 14040
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp sle i32 %inc, 14040
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_ri_sle
; CHECK: bdnz
; a < b
define void @test_pos2_ri_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, 13710
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp sle i32 %inc, 13710
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_ri_sle
; CHECK: bdnz
; a < b
define void @test_pos4_ri_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, 9920
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp sle i32 %inc, 9920
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_ri_sle
; CHECK: bdnz
; a < b
define void @test_pos8_ri_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, 18924
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp sle i32 %inc, 18924
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_ri_sle
; CHECK: bdnz
; a < b
define void @test_pos16_ri_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, 11812
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp sle i32 %inc, 11812
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos1_rr_sle
; FIXME: Support this loop!
; CHECK-NOT: bdnz
; a < b
define void @test_pos1_rr_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_rr_sle
; FIXME: Support this loop!
; CHECK-NOT: bdnz
; a < b
define void @test_pos2_rr_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_rr_sle
; FIXME: Support this loop!
; CHECK-NOT: bdnz
; a < b
define void @test_pos4_rr_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_rr_sle
; FIXME: Support this loop!
; CHECK-NOT: bdnz
; a < b
define void @test_pos8_rr_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_rr_sle
; FIXME: Support this loop!
; CHECK-NOT: bdnz
; a < b
define void @test_pos16_rr_sle(i8* nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp sle i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %i.04
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp sle i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

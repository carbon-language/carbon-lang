; RUN: llc -march=hexagon -hexagon-hwloop-preheader < %s | FileCheck %s

; Generate hardware loops when we also need to add a new preheader.
; we should generate two hardware loops for this test case.

; CHECK: loop0
; CHECK: endloop0
; CHECK: loop0
; CHECK: endloop0

@g = external global i32

define void @test(i32* nocapture %a, i32* nocapture %b, i32 %n) nounwind {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %for.body4.preheader, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %arrayidx.phi = phi i32* [ %arrayidx.inc, %for.body ], [ %a, %for.body.preheader ]
  %i.014 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %0 = load i32, i32* @g, align 4
  store i32 %0, i32* %arrayidx.phi, align 4
  %inc = add nsw i32 %i.014, 1
  %exitcond15 = icmp eq i32 %inc, 3
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond15, label %for.body4.preheader.loopexit, label %for.body

for.body4.preheader.loopexit:
  br label %for.body4.preheader

for.body4.preheader:
  br label %for.body4

for.body4:
  %arrayidx5.phi = phi i32* [ %arrayidx5.inc, %for.body4 ], [ %b, %for.body4.preheader ]
  %i1.013 = phi i32 [ %inc7, %for.body4 ], [ 0, %for.body4.preheader ]
  %1 = load i32, i32* @g, align 4
  store i32 %1, i32* %arrayidx5.phi, align 4
  %inc7 = add nsw i32 %i1.013, 1
  %exitcond = icmp eq i32 %inc7, 3
  %arrayidx5.inc = getelementptr i32, i32* %arrayidx5.phi, i32 1
  br i1 %exitcond, label %for.end8, label %for.body4

for.end8:
  ret void
}

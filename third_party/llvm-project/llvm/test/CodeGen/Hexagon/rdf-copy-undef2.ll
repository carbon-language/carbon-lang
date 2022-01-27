; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0
declare signext i16 @cat(i16 signext) #1
declare void @danny(i16 signext, i16 signext, i16 signext, i16* nocapture readonly, i16 signext, i16* nocapture) #1
declare void @sammy(i16* nocapture readonly, i16* nocapture readonly, i16* nocapture readonly, i32* nocapture, i16* nocapture, i16 signext, i16 signext, i16 signext) #1
declare i8* @llvm.stacksave() #2
declare void @llvm.stackrestore(i8*) #2

define i32 @fred(i16 signext %p0, i16 signext %p1, i16* nocapture readonly %p2, i16 signext %p3, i16* nocapture readonly %p4, i16* nocapture %p5) #1 {
entry:
  %0 = tail call i8* @llvm.stacksave()
  %vla = alloca i16, i32 undef, align 8
  %call17 = call signext i16 @cat(i16 signext 1) #1
  br i1 undef, label %for.cond23.preheader, label %for.end47

for.cond23.preheader:                             ; preds = %for.end40, %entry
  %i.190 = phi i16 [ %inc46, %for.end40 ], [ 0, %entry ]
  br i1 undef, label %for.body27, label %for.end40

for.body27:                                       ; preds = %for.body27, %for.cond23.preheader
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body27 ], [ 0, %for.cond23.preheader ]
  %call30 = call signext i16 @cat(i16 signext 7) #1
  %arrayidx32 = getelementptr inbounds i16, i16* %vla, i32 %indvars.iv
  store i16 %call30, i16* %arrayidx32, align 2
  %arrayidx37 = getelementptr inbounds i16, i16* undef, i32 %indvars.iv
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i16 undef, %p3
  br i1 %exitcond, label %for.end40, label %for.body27

for.end40:                                        ; preds = %for.body27, %for.cond23.preheader
  call void @sammy(i16* nonnull undef, i16* undef, i16* %p4, i32* null, i16* undef, i16 signext undef, i16 signext undef, i16 signext undef) #1
  %inc46 = add nuw nsw i16 %i.190, 1
  %exitcond94 = icmp eq i16 %inc46, %call17
  br i1 %exitcond94, label %for.end47.loopexit, label %for.cond23.preheader

for.end47.loopexit:                               ; preds = %for.end40
  %.pre = load i16, i16* undef, align 2
  br label %for.end47

for.end47:                                        ; preds = %for.end47.loopexit, %entry
  %1 = phi i16 [ %.pre, %for.end47.loopexit ], [ 0, %entry ]
  call void @danny(i16 signext %1, i16 signext %p0, i16 signext %p1, i16* %p2, i16 signext %p3, i16* %p5) #1
  call void @llvm.stackrestore(i8* %0)
  ret i32 undef
}


attributes #0 = { argmemonly nounwind }
attributes #1 = { optsize }
attributes #2 = { nounwind }

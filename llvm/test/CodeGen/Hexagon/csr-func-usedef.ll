; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

declare i8* @llvm.hexagon.circ.ldb(i8*, i8*, i32, i32) #1
declare i8* @llvm.hexagon.circ.stb(i8*, i32, i32, i32) #1

define zeroext i8 @circular_loop_test10(i8* %A, i8* %B, i32 %x, i32 %y, i32 %z, i32 %w) #0 {
entry:
  %element_load0 = alloca i8, align 1
  %element_load2 = alloca i8, align 1
  %element_load3 = alloca i8, align 1
  %element_load5 = alloca i8, align 1
  %or = or i32 %x, 100663296
  %or5 = or i32 %z, 100663296
  %or7 = or i32 %w, 100663296
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %p0.082 = phi i8* [ %A, %entry ], [ undef, %for.body ]
  %element_load.080 = phi i32 [ 0, %entry ], [ %add18, %for.body ]
  %p1.079 = phi i8* [ %B, %entry ], [ %1, %for.body ]
  %p2.078 = phi i8* [ undef, %entry ], [ %3, %for.body ]
  %p3.077 = phi i8* [ undef, %entry ], [ %4, %for.body ]
  %0 = call i8* @llvm.hexagon.circ.ldb(i8* %p0.082, i8* nonnull %element_load0, i32 %or, i32 2)
  %1 = call i8* @llvm.hexagon.circ.ldb(i8* %p1.079, i8* nonnull null, i32 0, i32 1)
  %2 = call i8* @llvm.hexagon.circ.ldb(i8* %p2.078, i8* nonnull %element_load2, i32 %or5, i32 3)
  %3 = call i8* @llvm.hexagon.circ.ldb(i8* %2, i8* nonnull %element_load5, i32 %or5, i32 1)
  %4 = call i8* @llvm.hexagon.circ.ldb(i8* %p3.077, i8* nonnull %element_load3, i32 %or7, i32 1)
  %5 = load i8, i8* null, align 1
  %conv = zext i8 %5 to i32
  %6 = load i8, i8* %element_load2, align 1
  %conv8 = zext i8 %6 to i32
  %7 = load i8, i8* %element_load3, align 1
  %conv9 = zext i8 %7 to i32
  %8 = load i8, i8* undef, align 1
  %conv11 = zext i8 %8 to i32
  %9 = load i8, i8* %element_load5, align 1
  %conv13 = zext i8 %9 to i32
  %10 = load i8, i8* %element_load0, align 1
  %conv15 = zext i8 %10 to i32
  %conv17 = and i32 %element_load.080, 255
  %add = add nuw nsw i32 %conv, %conv17
  %add10 = add nuw nsw i32 %add, %conv8
  %add12 = add nuw nsw i32 %add10, %conv9
  %add14 = add nuw nsw i32 %add12, %conv11
  %add16 = add nuw nsw i32 %add14, %conv13
  %add18 = add nuw nsw i32 %add16, %conv15
  %exitcond84 = icmp eq i32 undef, 200
  br i1 %exitcond84, label %for.body23, label %for.body

for.body23:                                       ; preds = %for.body23, %for.body
  %11 = call i8* @llvm.hexagon.circ.stb(i8* undef, i32 undef, i32 %or, i32 3)
  br i1 undef, label %for.body34, label %for.body23

for.body34:                                       ; preds = %for.body34, %for.body23
  %element_load.173 = phi i32 [ %add38, %for.body34 ], [ %add18, %for.body23 ]
  %arrayidx35 = getelementptr inbounds i8, i8* %B, i32 0
  %12 = load i8, i8* %arrayidx35, align 1
  %conv36 = zext i8 %12 to i32
  %conv37 = and i32 %element_load.173, 255
  %add38 = add nuw nsw i32 %conv36, %conv37
  br i1 undef, label %for.end42, label %for.body34

for.end42:                                        ; preds = %for.body34
  %conv39 = trunc i32 %add38 to i8
  ret i8 %conv39
}

attributes #0 = { nounwind optsize }
attributes #1 = { argmemonly nounwind }

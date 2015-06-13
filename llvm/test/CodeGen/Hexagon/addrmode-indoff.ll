; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Bug 6840. Use absolute+index addressing.

@ga = common global [1024 x i8] zeroinitializer, align 8
@gb = common global [1024 x i8] zeroinitializer, align 8

; CHECK: memub(r{{[0-9]+}}{{ *}}<<{{ *}}#0{{ *}}+{{ *}}##ga)
define zeroext i8 @lf2(i32 %i) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %i
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#0{{ *}}+{{ *}}##gb)
define signext i8 @lf2s(i32 %i) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @gb, i32 0, i32 %i
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK: memub(r{{[0-9]+}}{{ *}}<<{{ *}}#2{{ *}}+{{ *}}##ga)
define zeroext i8 @lf3(i32 %i) nounwind readonly {
entry:
  %mul = shl nsw i32 %i, 2
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#2{{ *}}+{{ *}}##gb)
define signext i8 @lf3s(i32 %i) nounwind readonly {
entry:
  %mul = shl nsw i32 %i, 2
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @gb, i32 0, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  ret i8 %0
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#0{{ *}}+{{ *}}##ga)
define void @sf4(i32 %i, i8 zeroext %j) nounwind {
entry:
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %i
  store i8 %j, i8* %arrayidx, align 1
  ret void
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#0{{ *}}+{{ *}}##gb)
define void @sf4s(i32 %i, i8 signext %j) nounwind {
entry:
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @gb, i32 0, i32 %i
  store i8 %j, i8* %arrayidx, align 1
  ret void
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#2{{ *}}+{{ *}}##ga)
define void @sf5(i32 %i, i8 zeroext %j) nounwind {
entry:
  %mul = shl nsw i32 %i, 2
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @ga, i32 0, i32 %mul
  store i8 %j, i8* %arrayidx, align 1
  ret void
}

; CHECK: memb(r{{[0-9]+}}{{ *}}<<{{ *}}#2{{ *}}+{{ *}}##gb)
define void @sf5s(i32 %i, i8 signext %j) nounwind {
entry:
  %mul = shl nsw i32 %i, 2
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @gb, i32 0, i32 %mul
  store i8 %j, i8* %arrayidx, align 1
  ret void
}

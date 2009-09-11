; RUN: opt < %s -instcombine -S | FileCheck %s

; Constant folding should fix notionally out-of-bounds indices
; and add inbounds keywords.

%struct.X = type { [3 x i32], [3 x i32] }

@Y = internal global [3 x %struct.X] zeroinitializer

define void @frob() {
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 1), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 2), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 0), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 3), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 4), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 2), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 5), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 6), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 7), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 8), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 0), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 9), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 10), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 2), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 11), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 12), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 13), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 14), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 15), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 1), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 16), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 17), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X]* @Y, i64 1, i64 0, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 18), align 8
; CHECK: store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 2, i64 0, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 36), align 8
; CHECK: store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 1, i64 0, i32 0, i64 1), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 19), align 8
  ret void
}

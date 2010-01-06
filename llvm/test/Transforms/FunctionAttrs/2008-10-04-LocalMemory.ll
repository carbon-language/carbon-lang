; RUN: opt < %s -functionattrs -S | grep readonly | count 3

%struct.X = type { i32*, i32* }

declare i32 @g(i32*) readonly

define i32 @f() {
	%x = alloca i32		; <i32*> [#uses=2]
	store i32 0, i32* %x
	%y = call i32 @g(i32* %x)		; <i32> [#uses=1]
	ret i32 %y
}

define i32 @foo() nounwind {
entry:
  %y = alloca %struct.X                           ; <%struct.X*> [#uses=2]
  %x = alloca %struct.X                           ; <%struct.X*> [#uses=2]
  %j = alloca i32                                 ; <i32*> [#uses=2]
  %i = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 0, i32* %i, align 4
  store i32 1, i32* %j, align 4
  %0 = getelementptr inbounds %struct.X* %y, i32 0, i32 0 ; <i32**> [#uses=1]
  store i32* %i, i32** %0, align 8
  %1 = getelementptr inbounds %struct.X* %x, i32 0, i32 1 ; <i32**> [#uses=1]
  store i32* %j, i32** %1, align 8
  %x1 = bitcast %struct.X* %x to i8*              ; <i8*> [#uses=2]
  %y2 = bitcast %struct.X* %y to i8*              ; <i8*> [#uses=1]
  call void @llvm.memcpy.i64(i8* %x1, i8* %y2, i64 8, i32 1)
  %2 = bitcast i8* %x1 to i32**                   ; <i32**> [#uses=1]
  %3 = load i32** %2, align 8                     ; <i32*> [#uses=1]
  %4 = load i32* %3, align 4                      ; <i32> [#uses=1]
  br label %return

return:                                           ; preds = %entry
  ret i32 %4
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

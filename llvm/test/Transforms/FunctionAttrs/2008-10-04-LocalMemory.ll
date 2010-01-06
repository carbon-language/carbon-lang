; RUN: opt < %s -functionattrs -S | FileCheck %s

%struct.X = type { i32*, i32* }

declare i32 @g(i32*) readnone

define i32 @f() {
; CHECK: @f() readnone
	%x = alloca i32		; <i32*> [#uses=2]
	store i32 0, i32* %x
	%y = call i32 @g(i32* %x)		; <i32> [#uses=1]
	ret i32 %y
}

define i32 @foo() nounwind {
; CHECK: @foo() nounwind readonly
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

define i32 @t(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK: @t(i32 %a, i32 %b, i32 %c) nounwind readnone
entry:
  %a.addr = alloca i32                            ; <i32*> [#uses=3]
  %c.addr = alloca i32                            ; <i32*> [#uses=2]
  store i32 %a, i32* %a.addr
  store i32 %c, i32* %c.addr
  %tmp = load i32* %a.addr                        ; <i32> [#uses=1]
  %tobool = icmp ne i32 %tmp, 0                   ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %p.0 = phi i32* [ %a.addr, %if.then ], [ %c.addr, %if.else ] ; <i32*> [#uses=1]
  %tmp2 = load i32* %p.0                          ; <i32> [#uses=1]
  ret i32 %tmp2
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@_ZTIc = external constant i8*
@_ZTId = external constant i8*
@_ZTIPKc = external constant i8*
@.str = private unnamed_addr constant [16 x i8] c"caught char %c\0A\00", align 1

define void @_Z3barv() uwtable optsize alwaysinline ssp {
entry:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %lpad

invoke.cont4:                                     ; preds = %lpad
  %eh.obj = extractvalue {i8*, i32} %exn, 0
  %tmp0 = tail call i8* @__cxa_begin_catch(i8* %eh.obj) nounwind
  %exn.scalar = load i8* %tmp0, align 1
  %conv = sext i8 %exn.scalar to i32
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([16 x i8]* @.str, i64 0, i64 0), i32 %conv) optsize
  tail call void @__cxa_end_catch() nounwind
  br label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont4
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
            catch i8** @_ZTIc
            filter i8** @_ZTIPKc
            catch i8** @_ZTId
  %tmp1 = extractvalue {i8*, i32} %exn, 1
  %tmp2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIc to i8*)) nounwind
  %tmp3 = icmp eq i32 %tmp1, %tmp2
  br i1 %tmp3, label %invoke.cont4, label %eh.resume

eh.resume:
  resume { i8*, i32 } %exn
}

declare void @_Z3quxv() optsize

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare void @llvm.eh.resume(i8*, i32)

declare i8* @__cxa_begin_catch(i8*)

declare i32 @printf(i8* nocapture, ...) nounwind optsize

declare void @__cxa_end_catch()

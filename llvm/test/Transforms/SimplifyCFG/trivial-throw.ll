; RUN: opt -simplifycfg -S < %s | FileCheck %s
; <rdar://problem/13360379>

@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS13TestException = linkonce_odr constant [16 x i8] c"13TestException\00"
@_ZTI13TestException = linkonce_odr unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([16 x i8]* @_ZTS13TestException, i32 0, i32 0) }

define void @throw(i32 %n) #0 {
entry:
  %exception = call i8* @__cxa_allocate_exception(i64 1) #4
  call void @__cxa_throw(i8* %exception, i8* bitcast ({ i8*, i8* }* @_ZTI13TestException to i8*), i8* null) #2
  unreachable
}

define void @func() #0 {
entry:
; CHECK: func()
; CHECK: invoke void @throw
; CHECK-NOT: call void @throw
  invoke void @throw(i32 42) #0
          to label %exit unwind label %lpad

lpad:
  %tmp0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  resume { i8*, i32 } %tmp0

exit:
  invoke void @abort() #2
          to label %invoke.cont unwind label %lpad1

invoke.cont:
  unreachable

lpad1:
  %tmp1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast ({ i8*, i8* }* @_ZTI13TestException to i8*)
  %tmp2 = extractvalue { i8*, i32 } %tmp1, 1
  %tmp3 = call i32 @llvm.eh.typeid.for(i8* bitcast ({ i8*, i8* }* @_ZTI13TestException to i8*)) #4
  %matches = icmp eq i32 %tmp2, %tmp3
  br i1 %matches, label %catch, label %eh.resume

catch:
  ret void

eh.resume:
  resume { i8*, i32 } %tmp1
}

define linkonce_odr hidden void @__clang_call_terminate(i8*) #1 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #4
  call void @_ZSt9terminatev() #5
  unreachable
}

declare void @abort() #2

declare i32 @llvm.eh.typeid.for(i8*) #3

declare void @__cxa_end_catch()

declare i8* @__cxa_allocate_exception(i64)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

attributes #0 = { ssp uwtable }
attributes #1 = { noinline noreturn nounwind }
attributes #2 = { noreturn }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

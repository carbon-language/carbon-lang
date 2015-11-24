; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@val = global i32 0, align 4
@_ZTIi = external constant i8*
; CHECK: @__llvm_profile_name__Z3bari = private constant [7 x i8] c"_Z3bari"
; CHECK: @__llvm_profile_name__Z3fooi = private constant [7 x i8] c"_Z3fooi"

define i32 @_Z3bari(i32 %i) {
entry:
  %rem = srem i32 %i, 3
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3bari, i32 0, i32 0), i64 23319865734, i32 2, i32 1)
  %exception = call i8* @__cxa_allocate_exception(i64 4)
  %tmp = bitcast i8* %exception to i32*
  store i32 %i, i32* %tmp, align 16
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable

if.end:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3bari, i32 0, i32 0), i64 23319865734, i32 2, i32 0)
  ret i32 0
}

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

define i32 @_Z3fooi(i32 %i) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %rem = srem i32 %i, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %mul = mul nsw i32 %i, 7
  %call = invoke i32 @_Z3bari(i32 %mul)
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3fooi, i32 0, i32 0), i64 57451243578, i32 4, i32 1)
  br label %if.end

lpad:
  %tmp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp1 = extractvalue { i8*, i32 } %tmp, 0
  %tmp2 = extractvalue { i8*, i32 } %tmp, 1
  br label %catch.dispatch

catch.dispatch:
  %tmp3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %tmp2, %tmp3
  br i1 %matches, label %catch, label %eh.resume

catch:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3fooi, i32 0, i32 0), i64 57451243578, i32 4, i32 2)
  %tmp4 = call i8* @__cxa_begin_catch(i8* %tmp1)
  %tmp5 = bitcast i8* %tmp4 to i32*
  %tmp6 = load i32, i32* %tmp5, align 4
  %tmp7 = load i32, i32* @val, align 4
  %sub = sub nsw i32 %tmp7, %tmp6
  store i32 %sub, i32* @val, align 4
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 -1

if.end:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3fooi, i32 0, i32 0), i64 57451243578, i32 4, i32 0)
  %tmp8 = load i32, i32* @val, align 4
  %add = add nsw i32 %tmp8, %i
  store i32 %add, i32* @val, align 4
  br label %try.cont

eh.resume:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @__llvm_profile_name__Z3fooi, i32 0, i32 0), i64 57451243578, i32 4, i32 3)
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %tmp1, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %tmp2, 1
  resume { i8*, i32 } %lpad.val3
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()


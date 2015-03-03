; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; struct SomeData {
;   int a;
;   int b;
; };
; 
; void may_throw();
; void does_not_throw(int i);
; void dump(int *, int, SomeData&);
; 
; void test() {
;   int NumExceptions = 0;
;   int ExceptionVal[10];
;   SomeData Data = { 0, 0 };
; 
;   for (int i = 0; i < 10; ++i) {
;     try {
;       may_throw();
;       Data.a += i;
;     }
;     catch (int e) {
;       ExceptionVal[NumExceptions] = e;
;       ++NumExceptions;
;       if (e == i)
;         Data.b += e;
;       else
;         Data.a += e;
;     }
;     does_not_throw(NumExceptions);
;   }
;   dump(ExceptionVal, NumExceptions, Data);
; }
;
; Unlike the cppeh-frame-vars.ll test, this test was generated using -O2
; optimization, which results in non-alloca values being used in the
; catch handler.

; ModuleID = 'cppeh-frame-vars.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%struct.SomeData = type { i32, i32 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

; This structure should be declared for the frame allocation block.
; CHECK: %"struct.\01?test@@YAXXZ.ehdata" = type { i32, i8*, i32, [10 x i32], i32, i32*, i32* }

; The function entry should be rewritten like this.
; CHECK: define void @"\01?test@@YAXXZ"() #0 {
; CHECK: entry:
; CHECK:  %frame.alloc = call i8* @llvm.frameallocate(i32 80)
; CHECK:  %eh.data = bitcast i8* %frame.alloc to %"struct.\01?test@@YAXXZ.ehdata"*
; CHECK-NOT:  %ExceptionVal = alloca [10 x i32], align 16
; CHECK:  %NumExceptions.020.reg2mem = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 2
; CHECK:  %i.019.reg2mem = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 4
; CHECK:  %ExceptionVal = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 3
; CHECK:  %Data = alloca i64, align 8
; CHECK:  %tmpcast = bitcast i64* %Data to %struct.SomeData*
; CHECK:  %0 = bitcast [10 x i32]* %ExceptionVal to i8*
; CHECK:  call void @llvm.lifetime.start(i64 40, i8* %0) #1
; CHECK:  store i64 0, i64* %Data, align 8
; CHECK:  %a.reg2mem = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 5
; CHECK:  %a = bitcast i64* %Data to i32*
; CHECK:  store i32* %a, i32** %a.reg2mem
; CHECK:  %b.reg2mem = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 6
; CHECK:  %b = getelementptr inbounds %struct.SomeData, %struct.SomeData* %tmpcast, i64 0, i32 1
; CHECK:  store i32* %b, i32** %b.reg2mem
; CHECK:  store i32 0, i32* %NumExceptions.020.reg2mem
; CHECK:  store i32 0, i32* %i.019.reg2mem
; CHECK:  br label %for.body

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 {
entry:
  %ExceptionVal = alloca [10 x i32], align 16
  %Data = alloca i64, align 8
  %tmpcast = bitcast i64* %Data to %struct.SomeData*
  %0 = bitcast [10 x i32]* %ExceptionVal to i8*
  call void @llvm.lifetime.start(i64 40, i8* %0) #1
  store i64 0, i64* %Data, align 8
  %a = bitcast i64* %Data to i32*
  %b = getelementptr inbounds %struct.SomeData, %struct.SomeData* %tmpcast, i64 0, i32 1
  br label %for.body

; CHECK: for.body:
; CHECK-NOT:  %NumExceptions.020 = phi i32 [ 0, %entry ], [ %NumExceptions.1, %try.cont ]
; CHECK-NOT:  %i.019 = phi i32 [ 0, %entry ], [ %inc5, %try.cont ]
; CHECK:  %i.019.reload = load i32, i32* %i.019.reg2mem
; CHECK:  %NumExceptions.020.reload = load i32, i32* %NumExceptions.020.reg2mem
for.body:                                         ; preds = %entry, %try.cont
  %NumExceptions.020 = phi i32 [ 0, %entry ], [ %NumExceptions.1, %try.cont ]
  %i.019 = phi i32 [ 0, %entry ], [ %inc5, %try.cont ]
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

; CHECK: invoke.cont:                                      ; preds = %for.body
; CHECK-NOT:  %1 = load i32, i32* %a, align 8, !tbaa !2
; CHECK-NOT:  %add = add nsw i32 %1, %i.019
; CHECK-NOT:  store i32 %add, i32* %a, align 8, !tbaa !2
; CHECK:   %a.reload3 = load volatile i32*, i32** %a.reg2mem
; CHECK:   %1 = load i32, i32* %a.reload3, align 8, !tbaa !2
; CHECK:   %add = add nsw i32 %1, %i.019.reload
; CHECK:   %a.reload2 = load volatile i32*, i32** %a.reg2mem
; CHECK:   store i32 %add, i32* %a.reload2, align 8, !tbaa !2
; CHECK:   br label %try.cont
invoke.cont:                                      ; preds = %for.body
  %1 = load i32, i32* %a, align 8, !tbaa !2
  %add = add nsw i32 %1, %i.019
  store i32 %add, i32* %a, align 8, !tbaa !2
  br label %try.cont

lpad:                                             ; preds = %for.body
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %3 = extractvalue { i8*, i32 } %2, 1
  %4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #1
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %5 = extractvalue { i8*, i32 } %2, 0
  %6 = tail call i8* @llvm.eh.begincatch(i8* %5) #1
  %7 = bitcast i8* %6 to i32*
  %8 = load i32, i32* %7, align 4, !tbaa !7
  %idxprom = sext i32 %NumExceptions.020 to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %ExceptionVal, i64 0, i64 %idxprom
  store i32 %8, i32* %arrayidx, align 4, !tbaa !7
  %inc = add nsw i32 %NumExceptions.020, 1
  %cmp1 = icmp eq i32 %8, %i.019
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %catch
  %9 = load i32, i32* %b, align 4, !tbaa !8
  %add2 = add nsw i32 %9, %i.019
  store i32 %add2, i32* %b, align 4, !tbaa !8
  br label %if.end

if.else:                                          ; preds = %catch
  %10 = load i32, i32* %a, align 8, !tbaa !2
  %add4 = add nsw i32 %10, %8
  store i32 %add4, i32* %a, align 8, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  tail call void @llvm.eh.endcatch() #1
  br label %try.cont

; CHECK: try.cont:                                         ; preds = %if.end, %invoke.cont
; CHECK-NOT:  %NumExceptions.1 = phi i32 [ %NumExceptions.020, %invoke.cont ], [ %inc, %if.end ]
; CHECK:   %NumExceptions.1 = phi i32 [ %NumExceptions.020.reload, %invoke.cont ], [ %inc, %if.end ]
; CHECK:   tail call void @"\01?does_not_throw@@YAXH@Z"(i32 %NumExceptions.1)
; CHECK-NOT:  %inc5 = add nuw nsw i32 %i.019, 1
; CHECK:   %inc5 = add nuw nsw i32 %i.019.reload, 1
; CHECK:   %cmp = icmp slt i32 %inc5, 10
; CHECK:   store i32 %NumExceptions.1, i32* %NumExceptions.020.reg2mem
; CHECK:   store i32 %inc5, i32* %i.019.reg2mem
; CHECK:   br i1 %cmp, label %for.body, label %for.end

try.cont:                                         ; preds = %if.end, %invoke.cont
  %NumExceptions.1 = phi i32 [ %NumExceptions.020, %invoke.cont ], [ %inc, %if.end ]
  tail call void @"\01?does_not_throw@@YAXH@Z"(i32 %NumExceptions.1)
  %inc5 = add nuw nsw i32 %i.019, 1
  %cmp = icmp slt i32 %inc5, 10
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %try.cont
  %NumExceptions.1.lcssa = phi i32 [ %NumExceptions.1, %try.cont ]
  %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %ExceptionVal, i64 0, i64 0
  call void @"\01?dump@@YAXPEAHHAEAUSomeData@@@Z"(i32* %arraydecay, i32 %NumExceptions.1.lcssa, %struct.SomeData* dereferenceable(8) %tmpcast)
  call void @llvm.lifetime.end(i64 40, i8* %0) #1
  ret void

eh.resume:                                        ; preds = %lpad
  %.lcssa = phi { i8*, i32 } [ %2, %lpad ]
  resume { i8*, i32 } %.lcssa
}

; The following catch handler should be outlined.
; CHECK: define i8* @"\01?test@@YAXXZ.catch"(i8*, i8*) {
; CHECK: catch.entry:
; CHECK:   %eh.alloc = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1)
; CHECK:   %eh.data = bitcast i8* %eh.alloc to %"struct.\01?test@@YAXXZ.ehdata"*
; CHECK:   %eh.obj.ptr = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 1
; CHECK:   %eh.obj = load i8*, i8** %eh.obj.ptr
; CHECK:   %eh.temp.alloca = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 2
; CHECK:   %NumExceptions.020.reload = load i32, i32* %eh.temp.alloca
; CHECK:   %ExceptionVal = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 3
; CHECK:   %eh.temp.alloca1 = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 4
; CHECK:   %i.019.reload = load i32, i32* %eh.temp.alloca1
; CHECK:   %eh.temp.alloca2 = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 5
; CHECK:   %a.reload = load i32*, i32** %eh.temp.alloca2
; CHECK:   %eh.temp.alloca3 = getelementptr inbounds %"struct.\01?test@@YAXXZ.ehdata", %"struct.\01?test@@YAXXZ.ehdata"* %eh.data, i32 0, i32 6
; CHECK:   %b.reload = load i32*, i32** %eh.temp.alloca3
; CHECK:   %2 = bitcast i8* %eh.obj to i32*
; CHECK:   %3 = load i32, i32* %2, align 4, !tbaa !7
; CHECK:   %idxprom = sext i32 %NumExceptions.020.reload to i64
; CHECK:   %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %ExceptionVal, i64 0, i64 %idxprom
; CHECK:   store i32 %3, i32* %arrayidx, align 4, !tbaa !7
; CHECK:   %inc = add nsw i32 %NumExceptions.020.reload, 1
; CHECK:   %cmp1 = icmp eq i32 %3, %i.019.reload
; CHECK:   br i1 %cmp1, label %if.then, label %if.else
;
; CHECK: if.then:                                          ; preds = %catch.entry
; CHECK:   %4 = load i32, i32* %b.reload, align 4, !tbaa !8
; CHECK:   %add2 = add nsw i32 %4, %i.019.reload
; CHECK:   store i32 %add2, i32* %b.reload, align 4, !tbaa !8
; CHECK:   br label %if.end
;
; CHECK: if.else:                                          ; preds = %catch.entry
; CHECK:   %5 = load i32, i32* %a.reload, align 8, !tbaa !2
; CHECK:   %add4 = add nsw i32 %5, %3
; CHECK:   store i32 %add4, i32* %a.reload, align 8, !tbaa !2
; CHECK:   br label %if.end
;
; CHECK: if.end:                                           ; preds = %if.else, %if.then
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont)
; CHECK: }

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @"\01?may_throw@@YAXXZ"() #2

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #3

declare i8* @llvm.eh.begincatch(i8*)

declare void @llvm.eh.endcatch()

declare void @"\01?does_not_throw@@YAXH@Z"(i32) #2

declare void @"\01?dump@@YAXPEAHHAEAUSomeData@@@Z"(i32*, i32, %struct.SomeData* dereferenceable(8)) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 228868)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"?AUSomeData@@", !4, i64 0, !4, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!4, !4, i64 0}
!8 = !{!3, !4, i64 4}

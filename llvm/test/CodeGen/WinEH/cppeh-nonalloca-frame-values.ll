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

; The function entry should be rewritten like this.
; CHECK: define void @"\01?test@@YAXXZ"() #0 {
; CHECK: entry:
; CHECK:   [[NUMEXCEPTIONS_REGMEM:\%.+]] = alloca i32
; CHECK:   [[I_REGMEM:\%.+]] = alloca i32
; CHECK:   [[E_PTR:\%.+]] = alloca i32, align 4
; CHECK:   [[EXCEPTIONVAL:\%.+]] = alloca [10 x i32], align 16
; CHECK:   [[DATA_PTR:\%.+]] = alloca i64, align 8
; CHECK:   [[TMPCAST:\%.+]] = bitcast i64* [[DATA_PTR]] to %struct.SomeData*
; CHECK:   [[TMP:\%.+]] = bitcast [10 x i32]* [[EXCEPTIONVAL]] to i8*
; CHECK:   call void @llvm.lifetime.start(i64 40, i8* [[TMP]])
; CHECK:   store i64 0, i64* [[DATA_PTR]], align 8
; CHECK:   [[A_REGMEM:\%.+]] = alloca i32*
; CHECK:   [[A_PTR:\%.+]] = bitcast i64* [[DATA_PTR]] to i32*
; CHECK:   store i32* [[A_PTR]], i32** [[A_REGMEM]]
; CHECK:   [[B_PTR:\%.+]] = getelementptr inbounds %struct.SomeData, %struct.SomeData* [[TMPCAST]], i64 0, i32 1
; CHECK:   [[B_REGMEM:\%.+]] = alloca i32*
; CHECK:   store i32* [[B_PTR]], i32** [[B_REGMEM]]
; CHECK:   store i32 0, i32* [[NUMEXCEPTIONS_REGMEM]]
; CHECK:   store i32 0, i32* [[I_REGMEM]]
; CHECK:   call void (...)* @llvm.frameescape(i32* %e, i32* %NumExceptions.020.reg2mem, [10 x i32]* [[EXCEPTIONVAL]], i32* [[I_REGMEM]], i32** [[A_REGMEM]], i32** [[B_REGMEM]])
; CHECK:   br label %for.body

; Function Attrs: uwtable
define void @"\01?test@@YAXXZ"() #0 {
entry:
  %e = alloca i32, align 4
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
; CHECK-NOT:  phi i32 [ 0, %entry ], [ {{\%NumExceptions.*}}, %try.cont ]
; CHECK-NOT:  phi i32 [ 0, %entry ], [ {{\%inc.*}}, %try.cont ]
; CHECK:  [[I_RELOAD:\%.+]] = load i32, i32* [[I_REGMEM]]
; CHECK:  [[NUMEXCEPTIONS_RELOAD:\%.+]] = load i32, i32* [[NUMEXCEPTIONS_REGMEM]]
for.body:                                         ; preds = %entry, %try.cont
  %NumExceptions.020 = phi i32 [ 0, %entry ], [ %NumExceptions.1, %try.cont ]
  %i.019 = phi i32 [ 0, %entry ], [ %inc5, %try.cont ]
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

; CHECK: invoke.cont:                                      ; preds = %for.body
; CHECK:   [[A_RELOAD:\%.+]] = load volatile i32*, i32** [[A_REGMEM]]
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[A_RELOAD]], align 8
; CHECK:   [[ADD:\%.+]] = add nsw i32 [[TMP1]], [[I_RELOAD]]
; CHECK:   [[A_RELOAD1:\%.+]] = load volatile i32*, i32** [[A_REGMEM]]
; CHECK:   store i32 [[ADD]], i32* [[A_RELOAD1]], align 8
; CHECK:   br label %try.cont
invoke.cont:                                      ; preds = %for.body
  %1 = load i32, i32* %a, align 8, !tbaa !2
  %add = add nsw i32 %1, %i.019
  store i32 %add, i32* %a, align 8, !tbaa !2
  br label %try.cont

; CHECK: [[LPAD_LABEL:lpad[0-9]*]]:{{[ ]+}}; preds = %for.body
; CHECK:   [[LPAD_VAL:\%.+]] = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
; CHECK:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK-NOT:   extractvalue { i8*, i32 }
; CHECK-NOT:   tail call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*))
; CHECK-NOT:   icmp eq i32
; CHECK-NOT:   br i1
; CHECK:   [[RECOVER:\%.+]] = call i8* (...)* @llvm.eh.actions({ i8*, i32 } [[LPAD_VAL]], i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32* %e, i8* bitcast (i8* (i8*, i8*)* @"\01?test@@YAXXZ.catch" to i8*))
; CHECK:   indirectbr i8* [[RECOVER]], [label %try.cont]

lpad:                                             ; preds = %for.body
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %3 = extractvalue { i8*, i32 } %2, 1
  %4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #1
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %eh.resume

; CHECK-NOT: catch:

catch:                                            ; preds = %lpad
  %5 = extractvalue { i8*, i32 } %2, 0
  %e.i8 = bitcast i32* %e to i8*
  call void @llvm.eh.begincatch(i8* %5, i8* %e.i8) #1
  %tmp8 = load i32, i32* %e, align 4, !tbaa !7
  %idxprom = sext i32 %NumExceptions.020 to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %ExceptionVal, i64 0, i64 %idxprom
  store i32 %tmp8, i32* %arrayidx, align 4, !tbaa !7
  %inc = add nsw i32 %NumExceptions.020, 1
  %cmp1 = icmp eq i32 %tmp8, %i.019
  br i1 %cmp1, label %if.then, label %if.else

; CHECK-NOT: if.then:

if.then:                                          ; preds = %catch
  %tmp9 = load i32, i32* %b, align 4, !tbaa !8
  %add2 = add nsw i32 %tmp9, %i.019
  store i32 %add2, i32* %b, align 4, !tbaa !8
  br label %if.end

; CHECK-NOT: if.else:

if.else:                                          ; preds = %catch
  %tmp10 = load i32, i32* %a, align 8, !tbaa !2
  %add4 = add nsw i32 %tmp10, %tmp8
  store i32 %add4, i32* %a, align 8, !tbaa !2
  br label %if.end

; CHECK-NOT: if.end:

if.end:                                           ; preds = %if.else, %if.then
  tail call void @llvm.eh.endcatch() #1
  br label %try.cont

; CHECK: try.cont:{{[ ]+}}; preds = %[[LPAD_LABEL]], %invoke.cont
; CHECK-NOT:  phi i32
; CHECK:   tail call void @"\01?does_not_throw@@YAXH@Z"(i32 [[NUMEXCEPTIONS_RELOAD]])
; CHECK:   [[INC:\%.+]] = add nuw nsw i32 [[I_RELOAD]], 1
; CHECK:   [[CMP:\%.+]] = icmp slt i32 [[INC]], 10
; CHECK:   store i32 [[NUMEXCEPTIONS_RELOAD]], i32* [[NUMEXCEPTIONS_REGMEM]]
; CHECK:   store i32 [[INC]], i32* [[I_REGMEM]]
; CHECK:   br i1 [[CMP]], label %for.body, label %for.end

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
; CHECK: define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*) {
; CHECK: entry:
; CHECK:   [[RECOVER_E:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
; CHECK:   [[E_PTR:\%.+]] = bitcast i8* [[RECOVER_E]] to i32*
; CHECK:   [[RECOVER_EH_TEMP:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 1)
; CHECK:   [[EH_TEMP:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP]] to i32*
; CHECK:   [[NUMEXCEPTIONS_RELOAD:\%.+]] = load i32, i32* [[EH_TEMP]]
; CHECK:   [[RECOVER_EXCEPTIONVAL:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 2)
; CHECK:   [[EXCEPTIONVAL:\%.+]] = bitcast i8* [[RECOVER_EXCEPTIONVAL]] to [10 x i32]*
; CHECK:   [[RECOVER_EH_TEMP1:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 3)
; CHECK:   [[EH_TEMP1:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP1]] to i32*
; CHECK:   [[I_RELOAD:\%.+]] = load i32, i32* [[EH_TEMP1]]
; CHECK:   [[RECOVER_EH_TEMP2:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 4)
; CHECK:   [[EH_TEMP2:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP2]] to i32**
; CHECK:   [[A_RELOAD:\%.+]] = load i32*, i32** [[EH_TEMP2]]
; CHECK:   [[RECOVER_EH_TEMP3:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 5)
; CHECK:   [[EH_TEMP3:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP3]] to i32**
; CHECK:   [[B_RELOAD:\%.+]] = load i32*, i32** [[EH_TEMP3]]
; CHECK:   [[E_I8PTR:\%.+]] = bitcast i32* [[E_PTR]] to i8*
; CHECK:   [[TMP:\%.+]] = load i32, i32* [[E_PTR]], align 4
; CHECK:   [[IDXPROM:\%.+]] = sext i32 [[NUMEXCEPTIONS_RELOAD]] to i64
; CHECK:   [[ARRAYIDX:\%.+]] = getelementptr inbounds [10 x i32], [10 x i32]* [[EXCEPTIONVAL]], i64 0, i64 [[IDXPROM]]
; CHECK:   store i32 [[TMP]], i32* [[ARRAYIDX]], align 4
; CHECK:   [[INC:\%.+]] = add nsw i32 [[NUMEXCEPTIONS_RELOAD]], 1
; CHECK:   [[CMP:\%.+]] = icmp eq i32 [[TMP]], [[I_RELOAD]]
; CHECK:   br i1 [[CMP]], label %if.then, label %if.else
;
; CHECK: if.then:{{[ ]+}}; preds = %entry
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[B_RELOAD]], align 4
; CHECK:   [[ADD:\%.+]] = add nsw i32 [[TMP1]], [[I_RELOAD]]
; CHECK:   store i32 [[ADD]], i32* [[B_RELOAD]], align 4
; CHECK:   br label %if.end
;
; CHECK: if.else:{{[ ]+}}; preds = %entry
; CHECK:   [[TMP2:\%.+]] = load i32, i32* [[A_RELOAD]], align 8
; CHECK:   [[ADD2:\%.+]] = add nsw i32 [[TMP2]], [[TMP]]
; CHECK:   store i32 [[ADD2]], i32* [[A_RELOAD]], align 8
; CHECK:   br label %if.end
;
; CHECK: if.end:{{[ ]+}}; preds = %if.else, %if.then
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont)
; CHECK: }

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @"\01?may_throw@@YAXXZ"() #2

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #3

declare void @llvm.eh.begincatch(i8*, i8*)

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

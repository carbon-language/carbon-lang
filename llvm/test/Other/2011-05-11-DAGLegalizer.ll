; RUN: llc < %s
; rdar://problem/9416774

; ModuleID = 'reduced.ii'
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.6"

%struct.GlAidRequest = type { [4 x %struct.GlOrbitInfoPerAidSource] }
%struct.GlOrbitInfoPerAidSource = type { [78 x i32] }

@kCLLogGenericComponent = external constant i8*
@__PRETTY_FUNCTION__._ZN12CLGllRequest13logAssistDataERK12GlAidRequest = private unnamed_addr constant [62 x i8] c"static void CLGllRequest::logAssistData(const GlAidRequest &)\00"
@.str = private unnamed_addr constant [75 x i8] c"\09GlOrbitInfoPerAidSource:,BE:0x%08lx,ALM:0x%08lx,LTO:0x%08lx,CBEE:0x%08lx\0A\00"

define void @_ZN12CLGllRequest13logAssistDataERK12GlAidRequest(%struct.GlAidRequest* %aidData) ssp align 2 {
entry:
  %aidData.addr = alloca %struct.GlAidRequest*, align 4
  %agg.tmp = alloca %struct.GlOrbitInfoPerAidSource, align 4
  %agg.tmp4 = alloca %struct.GlOrbitInfoPerAidSource, align 4
  %agg.tmp10 = alloca %struct.GlOrbitInfoPerAidSource, align 4
  %agg.tmp16 = alloca %struct.GlOrbitInfoPerAidSource, align 4
  store %struct.GlAidRequest* %aidData, %struct.GlAidRequest** %aidData.addr, align 4
  br label %do.body

do.body:                                          ; preds = %entry
  %tmp = load i8** @kCLLogGenericComponent, align 4
  %tmp1 = load %struct.GlAidRequest** %aidData.addr
  %eph = getelementptr inbounds %struct.GlAidRequest* %tmp1, i32 0, i32 0
  %arrayidx = getelementptr inbounds [4 x %struct.GlOrbitInfoPerAidSource]* %eph, i32 0, i32 0
  %tmp2 = bitcast %struct.GlOrbitInfoPerAidSource* %agg.tmp to i8*
  %tmp3 = bitcast %struct.GlOrbitInfoPerAidSource* %arrayidx to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp2, i8* %tmp3, i32 312, i32 4, i1 false)
  %tmp5 = load %struct.GlAidRequest** %aidData.addr
  %eph6 = getelementptr inbounds %struct.GlAidRequest* %tmp5, i32 0, i32 0
  %arrayidx7 = getelementptr inbounds [4 x %struct.GlOrbitInfoPerAidSource]* %eph6, i32 0, i32 1
  %tmp8 = bitcast %struct.GlOrbitInfoPerAidSource* %agg.tmp4 to i8*
  %tmp9 = bitcast %struct.GlOrbitInfoPerAidSource* %arrayidx7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp8, i8* %tmp9, i32 312, i32 4, i1 false)
  %tmp11 = load %struct.GlAidRequest** %aidData.addr
  %eph12 = getelementptr inbounds %struct.GlAidRequest* %tmp11, i32 0, i32 0
  %arrayidx13 = getelementptr inbounds [4 x %struct.GlOrbitInfoPerAidSource]* %eph12, i32 0, i32 2
  %tmp14 = bitcast %struct.GlOrbitInfoPerAidSource* %agg.tmp10 to i8*
  %tmp15 = bitcast %struct.GlOrbitInfoPerAidSource* %arrayidx13 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp14, i8* %tmp15, i32 312, i32 4, i1 false)
  %tmp17 = load %struct.GlAidRequest** %aidData.addr
  %eph18 = getelementptr inbounds %struct.GlAidRequest* %tmp17, i32 0, i32 0
  %arrayidx19 = getelementptr inbounds [4 x %struct.GlOrbitInfoPerAidSource]* %eph18, i32 0, i32 3
  %tmp20 = bitcast %struct.GlOrbitInfoPerAidSource* %agg.tmp16 to i8*
  %tmp21 = bitcast %struct.GlOrbitInfoPerAidSource* %arrayidx19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp20, i8* %tmp21, i32 312, i32 4, i1 false)
  call void (i8*, i32, i8*, i8*, ...)* @CLLoggingLog(i8* %tmp, i32 2, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__._ZN12CLGllRequest13logAssistDataERK12GlAidRequest, i32 0, i32 0), i8* getelementptr inbounds ([75 x i8]* @.str, i32 0, i32 0), %struct.GlOrbitInfoPerAidSource* byval %agg.tmp, %struct.GlOrbitInfoPerAidSource* byval %agg.tmp4, %struct.GlOrbitInfoPerAidSource* byval %agg.tmp10, %struct.GlOrbitInfoPerAidSource* byval %agg.tmp16)
  br label %do.end

do.end:                                           ; preds = %do.body
  ret void
}

declare void @CLLoggingLog(i8*, i32, i8*, i8*, ...)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

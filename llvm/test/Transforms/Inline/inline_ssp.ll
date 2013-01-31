; RUN: opt -inline %s -S | FileCheck %s
; Ensure SSP attributes are propagated correctly when inlining.

@.str = private unnamed_addr constant [11 x i8] c"fun_nossp\0A\00", align 1
@.str1 = private unnamed_addr constant [9 x i8] c"fun_ssp\0A\00", align 1
@.str2 = private unnamed_addr constant [15 x i8] c"fun_sspstrong\0A\00", align 1
@.str3 = private unnamed_addr constant [12 x i8] c"fun_sspreq\0A\00", align 1

; These first four functions (@fun_sspreq, @fun_sspstrong, @fun_ssp, @fun_nossp)
; are used by the remaining functions to ensure that the SSP attributes are
; propagated correctly.  The caller should have its SSP attribute set as:
; strictest(caller-ssp-attr, callee-ssp-attr), where strictness is ordered as:
;  sspreq > sspstrong > ssp > [no ssp]
define internal void @fun_sspreq() nounwind sspreq uwtable {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([12 x i8]* @.str3, i32 0, i32 0))
  ret void
}

define internal void @fun_sspstrong() nounwind sspstrong uwtable {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @.str2, i32 0, i32 0))
  ret void
}

define internal void @fun_ssp() nounwind ssp uwtable {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str1, i32 0, i32 0))
  ret void
}

define internal void @fun_nossp() nounwind uwtable {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0))
  ret void
}

; Tests start below 

define void @inline_req_req() nounwind sspreq uwtable {
entry:
; CHECK: @inline_req_req() nounwind sspreq uwtable 
  call void @fun_sspreq()
  ret void
}

define void @inline_req_strong() nounwind sspstrong uwtable {
entry:
; CHECK: @inline_req_strong() nounwind sspreq uwtable 
  call void @fun_sspreq()
  ret void
}

define void @inline_req_ssp() nounwind ssp uwtable {
entry:
; CHECK: @inline_req_ssp() nounwind sspreq uwtable 
  call void @fun_sspreq()
  ret void
}

define void @inline_req_nossp() nounwind uwtable {
entry:
; CHECK: @inline_req_nossp() nounwind sspreq uwtable 
  call void @fun_sspreq()
  ret void
}

define void @inline_strong_req() nounwind sspreq uwtable {
entry:
; CHECK: @inline_strong_req() nounwind sspreq uwtable 
  call void @fun_sspstrong()
  ret void
}


define void @inline_strong_strong() nounwind sspstrong uwtable {
entry:
; CHECK: @inline_strong_strong() nounwind sspstrong uwtable
  call void @fun_sspstrong()
  ret void
}

define void @inline_strong_ssp() nounwind ssp uwtable {
entry:
; CHECK: @inline_strong_ssp() nounwind sspstrong uwtable
  call void @fun_sspstrong()
  ret void
}

define void @inline_strong_nossp() nounwind uwtable {
entry:
; CHECK: @inline_strong_nossp() nounwind sspstrong uwtable
  call void @fun_sspstrong()
  ret void
}

define void @inline_ssp_req() nounwind sspreq uwtable {
entry:
; CHECK: @inline_ssp_req() nounwind sspreq uwtable
  call void @fun_ssp()
  ret void
}


define void @inline_ssp_strong() nounwind sspstrong uwtable {
entry:
; CHECK: @inline_ssp_strong() nounwind sspstrong uwtable
  call void @fun_ssp()
  ret void
}

define void @inline_ssp_ssp() nounwind ssp uwtable {
entry:
; CHECK: @inline_ssp_ssp() nounwind ssp uwtable
  call void @fun_ssp()
  ret void
}

define void @inline_ssp_nossp() nounwind uwtable {
entry:
; CHECK: @inline_ssp_nossp() nounwind ssp uwtable
  call void @fun_ssp()
  ret void
}

define void @inline_nossp_req() nounwind uwtable sspreq {
entry:
; CHECK: @inline_nossp_req() nounwind sspreq uwtable
  call void @fun_nossp()
  ret void
}


define void @inline_nossp_strong() nounwind sspstrong uwtable {
entry:
; CHECK: @inline_nossp_strong() nounwind sspstrong uwtable
  call void @fun_nossp()
  ret void
}

define void @inline_nossp_ssp() nounwind ssp uwtable {
entry:
; CHECK: @inline_nossp_ssp() nounwind ssp uwtable
  call void @fun_nossp()
  ret void
}

define void @inline_nossp_nossp() nounwind uwtable {
entry:
; CHECK: @inline_nossp_nossp() nounwind uwtable
  call void @fun_nossp()
  ret void
}

declare i32 @printf(i8*, ...)

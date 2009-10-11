; RUN: opt < %s -jump-threading -disable-output
; PR2285
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.system__secondary_stack__mark_id = type { i64, i64 }

define void @_ada_c35507b() {
entry:
	br label %bb

bb:		; preds = %bb13, %entry
	%ch.0 = phi i8 [ 0, %entry ], [ 0, %bb13 ]		; <i8> [#uses=1]
	%tmp11 = icmp ugt i8 %ch.0, 31		; <i1> [#uses=1]
	%tmp120 = call %struct.system__secondary_stack__mark_id @system__secondary_stack__ss_mark( )		; <%struct.system__secondary_stack__mark_id> [#uses=1]
	br i1 %tmp11, label %bb110, label %bb13

bb13:		; preds = %bb
	br label %bb

bb110:		; preds = %bb
	%mrv_gr124 = getresult %struct.system__secondary_stack__mark_id %tmp120, 1		; <i64> [#uses=0]
	unreachable
}

declare %struct.system__secondary_stack__mark_id @system__secondary_stack__ss_mark()



define fastcc void @findratio(double* nocapture %res1, double* nocapture %res2) nounwind ssp {
entry:
  br label %bb12

bb6.us:                                        
  %tmp = icmp eq i32 undef, undef              
  %tmp1 = fsub double undef, undef             
  %tmp2 = fcmp ult double %tmp1, 0.000000e+00  
  br i1 %tmp, label %bb6.us, label %bb13


bb12:                                            
  %tmp3 = fcmp ult double undef, 0.000000e+00  
  br label %bb13

bb13:                                            
  %.lcssa31 = phi double [ undef, %bb12 ], [ %tmp1, %bb6.us ]
  %.lcssa30 = phi i1 [ %tmp3, %bb12 ], [ %tmp2, %bb6.us ] 
  br i1 %.lcssa30, label %bb15, label %bb61

bb15:                                            
  %tmp4 = fsub double -0.000000e+00, %.lcssa31   
  ret void


bb61:                                            
  ret void
}

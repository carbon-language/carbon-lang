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


; PR5258
define i32 @test(i1 %cond, i1 %cond2, i32 %a) {
A:
  br i1 %cond, label %F, label %A1
F:
  br label %A1

A1:  
  %d = phi i1 [false, %A], [true, %F]
  %e = add i32 %a, %a
  br i1 %d, label %B, label %G
  
G:
  br i1 %cond2, label %B, label %D
  
B:
  %f = phi i32 [%e, %G], [%e, %A1]
  %b = add i32 0, 0
  switch i32 %a, label %C [
    i32 7, label %D
    i32 8, label %D
    i32 9, label %D
  ]

C:
  br label %D
  
D:
  %c = phi i32 [%e, %B], [%e, %B], [%e, %B], [%f, %C], [%e, %G]
  ret i32 %c
E:
  ret i32 412
}


define i32 @test2() nounwind {
entry:
        br i1 true, label %decDivideOp.exit, label %bb7.i

bb7.i:          ; preds = %bb7.i, %entry
        br label %bb7.i

decDivideOp.exit:               ; preds = %entry
        ret i32 undef
}


; PR3298

define i32 @test3(i32 %p_79, i32 %p_80) nounwind {
entry:
	br label %bb7

bb1:		; preds = %bb2
	br label %bb2

bb2:		; preds = %bb7, %bb1
	%l_82.0 = phi i8 [ 0, %bb1 ], [ %l_82.1, %bb7 ]		; <i8> [#uses=3]
	br i1 true, label %bb3, label %bb1

bb3:		; preds = %bb2
	%0 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %0, label %bb7, label %bb6

bb5:		; preds = %bb6
	%1 = icmp eq i8 %l_82.0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1.i, label %bb.i

bb.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

bb1.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

safe_div_func_char_s_s.exit:		; preds = %bb1.i, %bb.i
	br label %bb6

bb6:		; preds = %safe_div_func_char_s_s.exit, %bb3
	%p_80_addr.0 = phi i32 [ %p_80_addr.1, %bb3 ], [ 1, %safe_div_func_char_s_s.exit ]		; <i32> [#uses=2]
	%2 = icmp eq i32 %p_80_addr.0, 0		; <i1> [#uses=1]
	br i1 %2, label %bb7, label %bb5

bb7:		; preds = %bb6, %bb3, %entry
	%l_82.1 = phi i8 [ 1, %entry ], [ %l_82.0, %bb3 ], [ %l_82.0, %bb6 ]		; <i8> [#uses=2]
	%p_80_addr.1 = phi i32 [ 0, %entry ], [ %p_80_addr.1, %bb3 ], [ %p_80_addr.0, %bb6 ]		; <i32> [#uses=4]
	%3 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %3, label %bb8, label %bb2

bb8:		; preds = %bb7
	%4 = sext i8 %l_82.1 to i32		; <i32> [#uses=0]
	ret i32 0
}


; PR3353

define i32 @test4(i8 %X) {
entry:
        %Y = add i8 %X, 1
        %Z = add i8 %Y, 1
        br label %bb33.i

bb33.i:         ; preds = %bb33.i, %bb32.i
        switch i8 %Y, label %bb32.i [
                i8 39, label %bb35.split.i
                i8 13, label %bb33.i
        ]

bb35.split.i:
        ret i32 5
bb32.i:
        ret i32 1
}


define fastcc void @test5(i1 %tmp, i32 %tmp1) nounwind ssp {
entry:
  br i1 %tmp, label %bb12, label %bb13


bb12:                                            
  br label %bb13

bb13:                                            
  %.lcssa31 = phi i32 [ undef, %bb12 ], [ %tmp1, %entry ]
  %A = and i1 undef, undef
  br i1 %A, label %bb15, label %bb61

bb15:                                            
  ret void


bb61:                                            
  ret void
}


; PR5640
define fastcc void @test6(i1 %tmp, i1 %tmp1) nounwind ssp {
entry:
  br i1 %tmp, label %bb12, label %bb14

bb12:           
  br label %bb14

bb14:           
  %A = phi i1 [ %A, %bb13 ],  [ true, %bb12 ], [%tmp1, %entry]
  br label %bb13

bb13:                                            
  br i1 %A, label %bb14, label %bb61


bb61:                                            
  ret void
}


; PR5698
define void @test7(i32 %x) {
tailrecurse:
  switch i32 %x, label %return [
    i32 2, label %bb2
    i32 3, label %bb
  ]

bb:         
  switch i32 %x, label %return [
    i32 2, label %bb2
    i32 3, label %tailrecurse
  ]

bb2:        
  ret void

return:     
  ret void
}

; PR6119
define i32 @test8(i32 %action) nounwind {
entry:
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %cmp103 = xor i1 undef, undef                   ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}

; PR6119
define i32 @test9(i32 %action) nounwind {
entry:
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %0 = phi i1 [ undef, %lor.rhs ], [ true, %entry ] ; <i1> [#uses=1]
  %cmp103 = xor i1 undef, %0                      ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}

; PR6119
define i32 @test10(i32 %action, i32 %type) nounwind {
entry:
  %cmp2 = icmp eq i32 %type, 0                    ; <i1> [#uses=1]
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  %cmp101 = icmp eq i32 %action, 2                ; <i1> [#uses=1]
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %0 = phi i1 [ %cmp101, %lor.rhs ], [ true, %entry ] ; <i1> [#uses=1]
  %cmp103 = xor i1 %cmp2, %0                      ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}


; PR6305
define void @test11() nounwind {
entry:
  br label %A

A:                                             ; preds = %entry
  call void undef(i64 ptrtoint (i8* blockaddress(@test11, %A) to i64)) nounwind
  unreachable
}

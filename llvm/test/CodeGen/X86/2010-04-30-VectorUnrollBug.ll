; RUN: llc < %s -mtriple=x86_64-apple-darwin -disable-mmx | FileCheck %s
; <rdar://problem/7897988>

define void @test1(i8** %a, i64* %b, i64 %c, i64 %d) nounwind {
entry:
  %ptrtoarg37 = load i8** %a              ; <i8*> [#uses=1]
  %arglist1 = getelementptr i8** %a, i64 1 ; <i8**> [#uses=1]
  %ptrtoarg238 = load i8** %arglist1              ; <i8*> [#uses=1]
  %arglist4 = getelementptr i8** %a, i64 2 ; <i8**> [#uses=1]
  %ptrtoarg539 = load i8** %arglist4              ; <i8*> [#uses=1]
  %0 = load i64* %b                            ; <i64> [#uses=2]
  br label %loop.cond

loop.cond:                                        ; preds = %loop, %entry
  %iv = phi i64 [ %0, %entry ], [ %11, %loop ]    ; <i64> [#uses=3]
  %1 = icmp eq i64 %c, %iv                     ; <i1> [#uses=1]
  br i1 %1, label %return, label %loop

loop:                                             ; preds = %loop.cond
  %2 = bitcast i8* %ptrtoarg539 to i32 addrspace(1)* ; <i32 addrspace(1)*> [#uses=1]
  %3 = bitcast i8* %ptrtoarg238 to i32 addrspace(1)* ; <i32 addrspace(1)*> [#uses=1]
  %4 = bitcast i8* %ptrtoarg37 to i32 addrspace(1)* ; <i32 addrspace(1)*> [#uses=1]
  %tmp1.i = load i64* addrspace(256)* inttoptr (i64 248 to i64* addrspace(256)*) ; <i64*> [#uses=1]
  %tmp2.i = load i64* %tmp1.i                     ; <i64> [#uses=1]
  %conv.i = trunc i64 %tmp2.i to i32              ; <i32> [#uses=1]
  %conv1.i = sext i32 %conv.i to i64              ; <i64> [#uses=1]
  %5 = bitcast i32 addrspace(1)* %4 to i32*       ; <i32*> [#uses=2]
  %i_times_3 = mul i64 %conv1.i, 3                 ; <i64> [#uses=4]
  %inptrA.i = getelementptr inbounds i32* %5, i64 %i_times_3 ; <i32*> [#uses=1]
  %6 = bitcast i32* %inptrA.i to i64*        ; <i64*> [#uses=1]
  %i_times_3_plus_2 = add i64 %i_times_3, 2     ; <i64> [#uses=3]
  %A.xy = load i64* %6                    ; <i64> [#uses=1]
  %inptrA.ip2 = getelementptr inbounds i32* %5, i64 %i_times_3_plus_2 ; <i32*> [#uses=1]
  %A.xyuu = insertelement <2 x i64> undef, i64 %A.xy, i32 0 ; <<2 x i64>> [#uses=1]
  %A.xy__ = insertelement <2 x i64> %A.xyuu, i64 0, i32 1 ; <<2 x i64>> [#uses=1]
  %tmp13.i.i20 = bitcast <2 x i64> %A.xy__ to <4 x i32> ; <<4 x i32>> [#uses=1]
  %A.z = load i32* %inptrA.ip2      ; <i32> [#uses=1]

; The "movl" is the load of %A.z. The registers aren't important. Here are what they map to:
;
;   %rbx -> %i_times_3
;   %r9 ->  %5
;   %r9 + (%rbx + 2) * 4 -> %inptrA.ip2
;   8(%r9,%rbx,4) -> *%inptrA.ip2 -> %A.z
;
;      CHECK: sarl %cl, %r15d
; CHECK-NEXT: pinsrd    $0, %r15d, %xmm3
; CHECK-NEXT: pinsrd    $1, %r14d, %xmm3
; CHECK-NEXT: movl      8(%r9,%rbx,4), %r14d
; CHECK-NEXT: pextrd    $2, %xmm2, %ecx

  %A.xyz_ = insertelement <4 x i32> %tmp13.i.i20, i32 %A.z, i32 2 ; <<4 x i32>> [#uses=1]
  %A.xyz = shufflevector <4 x i32> %A.xyz_, <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2> ; <<3 x i32>> [#uses=1]
  %7 = bitcast i32 addrspace(1)* %3 to i32*       ; <i32*> [#uses=2]
  %add.ptr.i.i8 = getelementptr inbounds i32* %7, i64 %i_times_3 ; <i32*> [#uses=1]
  %8 = bitcast i32* %add.ptr.i.i8 to i64*         ; <i64*> [#uses=1]
  %B.xy = load i64* %8                      ; <i64> [#uses=1]
  %arrayidx15.i.i = getelementptr inbounds i32* %7, i64 %i_times_3_plus_2 ; <i32*> [#uses=1]
  %B.xyuu = insertelement <2 x i64> undef, i64 %B.xy, i32 0 ; <<2 x i64>> [#uses=1]
  %B.z = load i32* %arrayidx15.i.i          ; <i32> [#uses=1]
  %vecinit2.i.i.i = insertelement <2 x i64> %B.xyuu, i64 0, i32 1 ; <<2 x i64>> [#uses=1]
  %B.xy__ = bitcast <2 x i64> %vecinit2.i.i.i to <4 x i32> ; <<4 x i32>> [#uses=1]
  %B.xyz_ = insertelement <4 x i32> %B.xy__, i32 %B.z, i32 2 ; <<4 x i32>> [#uses=1]
  %B.xyz = shufflevector <4 x i32> %B.xyz_, <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2> ; <<3 x i32>> [#uses=1]
  %and.i = and <3 x i32> %B.xyz, <i32 31, i32 31, i32 31> ; <<3 x i32>> [#uses=1]
  %shr.i = ashr <3 x i32> %A.xyz, %and.i    ; <<3 x i32>> [#uses=2]
  %9 = bitcast i32 addrspace(1)* %2 to i32*       ; <i32*> [#uses=2]
  %tmp3.i.i = shufflevector <3 x i32> %shr.i, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef> ; <<4 x i32>> [#uses=1]
  %add.ptr.i.i = getelementptr inbounds i32* %9, i64 %i_times_3 ; <i32*> [#uses=1]
  %10 = bitcast i32* %add.ptr.i.i to i8*          ; <i8*> [#uses=1]
  call void @llvm.x86.sse2.storel.dq(i8* %10, <4 x i32> %tmp3.i.i) nounwind
  %vecext.i.i.i = extractelement <3 x i32> %shr.i, i32 2 ; <i32> [#uses=1]
  %arrayidx.i.i = getelementptr inbounds i32* %9, i64 %i_times_3_plus_2 ; <i32*> [#uses=1]
  store i32 %vecext.i.i.i, i32* %arrayidx.i.i
  %11 = add i64 %iv, %d                      ; <i64> [#uses=1]
  %tmp = add i64 %d, %iv                     ; <i64> [#uses=1]
  store i64 %tmp, i64* %b
  br label %loop.cond

return:                                           ; preds = %loop.cond
  store i64 %0, i64* %b
  ret void
}

declare void @llvm.x86.sse2.storel.dq(i8*, <4 x i32>) nounwind

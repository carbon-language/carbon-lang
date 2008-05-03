; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep psllq | grep 32
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep psllq | grep 32
; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep psrad

define i64 @t1(<1 x i64> %mm1) nounwind  {
entry:
	%tmp6 = tail call <1 x i64> @llvm.x86.mmx.pslli.q( <1 x i64> %mm1, i32 32 )		; <<1 x i64>> [#uses=1]
	%retval1112 = bitcast <1 x i64> %tmp6 to i64		; <i64> [#uses=1]
	ret i64 %retval1112
}

declare <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64>, i32) nounwind readnone 

define i64 @t2(<2 x i32> %mm1, <2 x i32> %mm2) nounwind  {
entry:
	%tmp7 = tail call <2 x i32> @llvm.x86.mmx.psra.d( <2 x i32> %mm1, <2 x i32> %mm2 ) nounwind readnone 		; <<2 x i32>> [#uses=1]
	%retval1112 = bitcast <2 x i32> %tmp7 to i64		; <i64> [#uses=1]
	ret i64 %retval1112
}

declare <2 x i32> @llvm.x86.mmx.psra.d(<2 x i32>, <2 x i32>) nounwind readnone 

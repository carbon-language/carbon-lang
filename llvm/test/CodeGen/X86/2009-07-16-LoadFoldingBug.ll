; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s

; CHECK: _foo:
; CHECK: pavgw LCPI1_4(%rip)

; rdar://7057804

define void @foo(i16* %out8x8, i16* %in8x8, i32 %lastrow) optsize ssp {
entry:
	%0 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518>, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=2]
	%1 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %0, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%2 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> zeroinitializer, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%3 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %2, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i10 = add <8 x i16> %0, %3		; <<8 x i16>> [#uses=1]
	%4 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> zeroinitializer, <8 x i16> %1) nounwind readnone		; <<8 x i16>> [#uses=1]
	%5 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i10, <8 x i16> %4) nounwind readnone		; <<8 x i16>> [#uses=3]
	%6 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %5, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%7 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518, i16 6518>, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=2]
	%8 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %7, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%9 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> zeroinitializer, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%10 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %9, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i8 = add <8 x i16> %7, %10		; <<8 x i16>> [#uses=1]
	%11 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> %8) nounwind readnone		; <<8 x i16>> [#uses=1]
	%12 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i8, <8 x i16> %11) nounwind readnone		; <<8 x i16>> [#uses=1]
	%13 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> undef, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%14 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %5, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%15 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %5, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%16 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %6, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%17 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %12, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=1]
	%18 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %13, <8 x i16> %15) nounwind readnone		; <<8 x i16>> [#uses=1]
	%19 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> %14) nounwind readnone		; <<8 x i16>> [#uses=2]
	%20 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=4]
	%21 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> %17) nounwind readnone		; <<8 x i16>> [#uses=1]
	%22 = bitcast <8 x i16> %21 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%23 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170>, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=2]
	%24 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %23, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%25 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> zeroinitializer, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%26 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %25, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i6 = add <8 x i16> %23, %26		; <<8 x i16>> [#uses=1]
	%27 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> %24) nounwind readnone		; <<8 x i16>> [#uses=1]
	%28 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i6, <8 x i16> %27) nounwind readnone		; <<8 x i16>> [#uses=1]
	%29 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170>, <8 x i16> undef) nounwind readnone		; <<8 x i16>> [#uses=2]
	%30 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %29, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%31 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> zeroinitializer, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%32 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %31, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i4 = add <8 x i16> %29, %32		; <<8 x i16>> [#uses=1]
	%33 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> undef, <8 x i16> %30) nounwind readnone		; <<8 x i16>> [#uses=1]
	%34 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i4, <8 x i16> %33) nounwind readnone		; <<8 x i16>> [#uses=1]
	%35 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170>, <8 x i16> %20) nounwind readnone		; <<8 x i16>> [#uses=2]
	%tmp.i2.i1 = mul <8 x i16> %20, <i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170, i16 23170>		; <<8 x i16>> [#uses=1]
	%36 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %35, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%37 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %tmp.i2.i1, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%38 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %37, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i2 = add <8 x i16> %35, %38		; <<8 x i16>> [#uses=1]
	%39 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %19, <8 x i16> %36) nounwind readnone		; <<8 x i16>> [#uses=1]
	%40 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i2, <8 x i16> %39) nounwind readnone		; <<8 x i16>> [#uses=1]
	%41 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> <i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170>, <8 x i16> %20) nounwind readnone		; <<8 x i16>> [#uses=2]
	%tmp.i2.i = mul <8 x i16> %20, <i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170>		; <<8 x i16>> [#uses=1]
	%42 = call <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16> %41, <8 x i16> <i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384, i16 16384>) nounwind readnone		; <<8 x i16>> [#uses=1]
	%43 = call <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %tmp.i2.i, i32 14) nounwind readnone		; <<8 x i16>> [#uses=1]
	%44 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %43, <8 x i16> zeroinitializer) nounwind readnone		; <<8 x i16>> [#uses=1]
	%tmp.i.i = add <8 x i16> %41, %44		; <<8 x i16>> [#uses=1]
	%45 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %19, <8 x i16> %42) nounwind readnone		; <<8 x i16>> [#uses=1]
	%46 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %tmp.i.i, <8 x i16> %45) nounwind readnone		; <<8 x i16>> [#uses=1]
	%47 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %18, <8 x i16> %16) nounwind readnone		; <<8 x i16>> [#uses=1]
	%48 = bitcast <8 x i16> %47 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%49 = bitcast <8 x i16> %28 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%50 = getelementptr i16* %out8x8, i64 8		; <i16*> [#uses=1]
	%51 = bitcast i16* %50 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %49, <2 x i64>* %51, align 16
	%52 = bitcast <8 x i16> %40 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%53 = getelementptr i16* %out8x8, i64 16		; <i16*> [#uses=1]
	%54 = bitcast i16* %53 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %52, <2 x i64>* %54, align 16
	%55 = getelementptr i16* %out8x8, i64 24		; <i16*> [#uses=1]
	%56 = bitcast i16* %55 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %48, <2 x i64>* %56, align 16
	%57 = bitcast <8 x i16> %46 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%58 = getelementptr i16* %out8x8, i64 40		; <i16*> [#uses=1]
	%59 = bitcast i16* %58 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %57, <2 x i64>* %59, align 16
	%60 = bitcast <8 x i16> %34 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%61 = getelementptr i16* %out8x8, i64 48		; <i16*> [#uses=1]
	%62 = bitcast i16* %61 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %60, <2 x i64>* %62, align 16
	%63 = getelementptr i16* %out8x8, i64 56		; <i16*> [#uses=1]
	%64 = bitcast i16* %63 to <2 x i64>*		; <<2 x i64>*> [#uses=1]
	store <2 x i64> %22, <2 x i64>* %64, align 16
	ret void
}

declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pcmpeq.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16>, i32) nounwind readnone

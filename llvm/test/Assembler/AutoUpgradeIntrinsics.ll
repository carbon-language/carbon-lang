; Tests to make sure intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | not grep {i32 @llvm\\.ct}
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep {llvm\\.part\\.set\\.i\[0-9\]*\\.i\[0-9\]*\\.i\[0-9\]*}
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep {llvm\\.part\\.select\\.i\[0-9\]*\\.i\[0-9\]*}
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep {llvm\\.bswap\\.i\[0-9\]*\\.i\[0-9\]*}
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep {llvm\\.x86\\.sse2\\.loadu}
; RUN: llvm-as < %s | llvm-dis | \
; RUN:   grep {llvm\\.x86\\.mmx\\.ps} | grep {x86_mmx} | count 16
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare i32 @llvm.ctpop.i28(i28 %val)
declare i32 @llvm.cttz.i29(i29 %val)
declare i32 @llvm.ctlz.i30(i30 %val)

define i32 @test_ct(i32 %A) {
  %c1 = call i32 @llvm.ctpop.i28(i28 1234)
  %c2 = call i32 @llvm.cttz.i29(i29 2345)
  %c3 = call i32 @llvm.ctlz.i30(i30 3456)
  %r1 = add i32 %c1, %c2
  %r2 = add i32 %r1, %c3
  ret i32 %r2
}

declare i32 @llvm.part.set.i32.i32.i32(i32 %x, i32 %rep, i32 %hi, i32 %lo)
declare i16 @llvm.part.set.i16.i16.i16(i16 %x, i16 %rep, i32 %hi, i32 %lo)
define i32 @test_part_set(i32 %A, i16 %B) {
  %a = call i32 @llvm.part.set.i32.i32.i32(i32 %A, i32 27, i32 8, i32 0)
  %b = call i16 @llvm.part.set.i16.i16.i16(i16 %B, i16 27, i32 8, i32 0)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}

declare i32 @llvm.part.select.i32.i32(i32 %x, i32 %hi, i32 %lo)
declare i16 @llvm.part.select.i16.i16(i16 %x, i32 %hi, i32 %lo)
define i32 @test_part_select(i32 %A, i16 %B) {
  %a = call i32 @llvm.part.select.i32.i32(i32 %A, i32 8, i32 0)
  %b = call i16 @llvm.part.select.i16.i16(i16 %B, i32 8, i32 0)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}

declare i32 @llvm.bswap.i32.i32(i32 %x)
declare i16 @llvm.bswap.i16.i16(i16 %x)
define i32 @test_bswap(i32 %A, i16 %B) {
  %a = call i32 @llvm.bswap.i32.i32(i32 %A)
  %b = call i16 @llvm.bswap.i16.i16(i16 %B)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}

declare <4 x i16> @llvm.x86.mmx.psra.w(<4 x i16>, <2 x i32>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.psll.w(<4 x i16>, <2 x i32>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.psrl.w(<4 x i16>, <2 x i32>) nounwind readnone 
define void @sh16(<4 x i16> %A, <2 x i32> %B) {
	%r1 = call <4 x i16> @llvm.x86.mmx.psra.w( <4 x i16> %A, <2 x i32> %B )		; <<4 x i16>> [#uses=0]
	%r2 = call <4 x i16> @llvm.x86.mmx.psll.w( <4 x i16> %A, <2 x i32> %B )		; <<4 x i16>> [#uses=0]
	%r3 = call <4 x i16> @llvm.x86.mmx.psrl.w( <4 x i16> %A, <2 x i32> %B )		; <<4 x i16>> [#uses=0]
	ret void
}

declare <2 x i32> @llvm.x86.mmx.psra.d(<2 x i32>, <2 x i32>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psll.d(<2 x i32>, <2 x i32>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psrl.d(<2 x i32>, <2 x i32>) nounwind readnone 
define void @sh32(<2 x i32> %A, <2 x i32> %B) {
	%r1 = call <2 x i32> @llvm.x86.mmx.psra.d( <2 x i32> %A, <2 x i32> %B )		; <<2 x i32>> [#uses=0]
	%r2 = call <2 x i32> @llvm.x86.mmx.psll.d( <2 x i32> %A, <2 x i32> %B )		; <<2 x i32>> [#uses=0]
	%r3 = call <2 x i32> @llvm.x86.mmx.psrl.d( <2 x i32> %A, <2 x i32> %B )		; <<2 x i32>> [#uses=0]
	ret void
}

declare <1 x i64> @llvm.x86.mmx.psll.q(<1 x i64>, <2 x i32>) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.psrl.q(<1 x i64>, <2 x i32>) nounwind readnone 
define void @sh64(<1 x i64> %A, <2 x i32> %B) {
	%r1 = call <1 x i64> @llvm.x86.mmx.psll.q( <1 x i64> %A, <2 x i32> %B )		; <<1 x i64>> [#uses=0]
	%r2 = call <1 x i64> @llvm.x86.mmx.psrl.q( <1 x i64> %A, <2 x i32> %B )		; <<1 x i64>> [#uses=0]
	ret void
}

declare <4 x float> @llvm.x86.sse.loadu.ps(i8*) nounwind readnone
declare <16 x i8> @llvm.x86.sse2.loadu.dq(i8*) nounwind readnone
declare <2 x double> @llvm.x86.sse2.loadu.pd(double*) nounwind readnone
define void @test_loadu(i8* %a, double* %b) {
  %v0 = call <4 x float> @llvm.x86.sse.loadu.ps(i8* %a)
  %v1 = call <16 x i8> @llvm.x86.sse2.loadu.dq(i8* %a)
  %v2 = call <2 x double> @llvm.x86.sse2.loadu.pd(double* %b)
  ret void
}

declare void @llvm.x86.sse.movnt.ps(i8*, <4 x float>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.dq(i8*, <2 x double>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.pd(i8*, <2 x double>) nounwind readnone 
declare void @llvm.x86.sse2.movnt.i(i8*, i32) nounwind readnone 

define void @f(<4 x float> %A, i8* %B, <2 x double> %C, i32 %D) {
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse.movnt.ps(i8* %B, <4 x float> %A)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.dq(i8* %B, <2 x double> %C)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.pd(i8* %B, <2 x double> %C)
; CHECK: store{{.*}}nontemporal
  call void @llvm.x86.sse2.movnt.i(i8* %B, i32 %D)
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32) nounwind

define void @p(i8* %ptr) {
; CHECK: llvm.prefetch(i8* %ptr, i32 0, i32 1, i32 1)
  tail call void @llvm.prefetch(i8* %ptr, i32 0, i32 1)
  ret void
}

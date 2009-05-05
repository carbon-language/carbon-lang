; Check that we do not strength reduce to an overflowiong loop condition
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep 4755160694556239428

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"
@"initialized$$$CFE_id_3d9e38d5_main" = linkonce global [3 x i64] [i64 6729544817907097974, i64 754394135718280480, i64 0]		
@"initialized$main$$CFE_id_3d9e38d5_main" = linkonce global [512 x i64] zeroinitializer		

define i32 @main() {
"file loop.c, line 10, bb14":
	%vals = alloca [512 x i64], align 16		
	br label %"file loop.c, line 10, bb101"

"file loop.c, line 10, bb101":		
	br label %"file loop.c, line 10, in inner vector loop at depth 0, bb62"

"file loop.c, line 10, in inner vector loop at depth 0, bb62":		
	%"$SI_S13.0" = phi i64 [ 0, %"file loop.c, line 10, bb101" ], [ %r35, %"file loop.c, line 13, in inner vector loop at depth 0, bb63" ]		
	%"$LC_S14.0" = phi i64 [ -256, %"file loop.c, line 10, bb101" ], [ %r37, %"file loop.c, line 13, in inner vector loop at depth 0, bb63" ]		
	br label %"file loop.c, line 13, in inner vector loop at depth 0, bb63"

"file loop.c, line 13, in inner vector loop at depth 0, bb63":		
	%r3 = add i64 %"$SI_S13.0", ptrtoint ([512 x i64]* @"initialized$main$$CFE_id_3d9e38d5_main" to i64)		
	%vals18 = bitcast [512 x i64]* %vals to i8*		
	%ctg2 = getelementptr i8* %vals18, i64 %"$SI_S13.0"		
	%r9 = inttoptr i64 %r3 to <2 x i64>*		
	%r10 = load <2 x i64>* %r9, align 8		
	%r12 = bitcast i8* %ctg2 to <2 x i64>*		
	store <2 x i64> %r10, <2 x i64>* %r12, align 16
	%tmp17 = inttoptr i64 %r3 to <2 x i64>*		
	%r15 = getelementptr <2 x i64>* %tmp17, i64 1		
	%r16 = load <2 x i64>* %r15, align 8		
	%ctg2.sum19 = or i64 %"$SI_S13.0", 16		
	%r19 = getelementptr i8* %vals18, i64 %ctg2.sum19		
	%0 = bitcast i8* %r19 to <2 x i64>*		
	store <2 x i64> %r16, <2 x i64>* %0, align 16
	%tmp15 = inttoptr i64 %r3 to <2 x i64>*		
	%r22 = getelementptr <2 x i64>* %tmp15, i64 2		
	%r23 = load <2 x i64>* %r22, align 8		
	%ctg2.sum20 = or i64 %"$SI_S13.0", 32		
	%r26 = getelementptr i8* %vals18, i64 %ctg2.sum20		
	%1 = bitcast i8* %r26 to <2 x i64>*		
	store <2 x i64> %r23, <2 x i64>* %1, align 16
	%tmp13 = inttoptr i64 %r3 to <2 x i64>*		
	%r29 = getelementptr <2 x i64>* %tmp13, i64 3		
	%r30 = load <2 x i64>* %r29, align 8		
	%ctg2.sum21 = or i64 %"$SI_S13.0", 48		
	%r33 = getelementptr i8* %vals18, i64 %ctg2.sum21		
	%2 = bitcast i8* %r33 to <2 x i64>*		
	store <2 x i64> %r30, <2 x i64>* %2, align 16
	%r35 = add i64 %"$SI_S13.0", 64		
	%r37 = add i64 %"$LC_S14.0", 4		
	%r39 = icmp slt i64 %r37, 0		
	br i1 %r39, label %"file loop.c, line 10, in inner vector loop at depth 0, bb62", label %"file loop.c, line 15, bb51"

"file loop.c, line 15, bb51":		
	%3 = getelementptr [512 x i64]* %vals, i64 0, i64 0		
	store i64 0, i64* %3, align 16
	br label %"file loop.c, line 10, in inner loop at depth 0, bb65"

"file loop.c, line 10, in inner loop at depth 0, bb65":		
	%"$vals_WR0_S15.0" = phi i64 [ 0, %"file loop.c, line 15, bb51" ], [ %r70, %"file loop.c, line 16, in inner loop at depth 0, bb67" ]		
	%"$i_S16.0" = phi i64 [ 0, %"file loop.c, line 15, bb51" ], [ %r78, %"file loop.c, line 16, in inner loop at depth 0, bb67" ]		
	br label %"file loop.c, line 16, in inner loop at depth 0, bb67"

"file loop.c, line 16, in inner loop at depth 0, bb67":		
	%r45 = add i64 %"$vals_WR0_S15.0", 81985529216486895		
	%r5111 = or i64 %"$i_S16.0", 1		
	%r5212 = getelementptr [512 x i64]* %vals, i64 0, i64 %r5111		
	store i64 %r45, i64* %r5212, align 8
	%r54 = add i64 %"$vals_WR0_S15.0", 163971058432973790		
	%r599 = or i64 %"$i_S16.0", 2		
	%r6010 = getelementptr [512 x i64]* %vals, i64 0, i64 %r599		
	store i64 %r54, i64* %r6010, align 16
	%r62 = add i64 %"$vals_WR0_S15.0", 245956587649460685		
	%r677 = or i64 %"$i_S16.0", 3		
	%r688 = getelementptr [512 x i64]* %vals, i64 0, i64 %r677		
	store i64 %r62, i64* %r688, align 8
	%r70 = add i64 %"$vals_WR0_S15.0", 327942116865947580		
	%r75 = add i64 %"$i_S16.0", 4		
	%r766 = getelementptr [512 x i64]* %vals, i64 0, i64 %r75		
	store i64 %r70, i64* %r766, align 16
	%r78 = add i64 %"$i_S16.0", 4		
	%r80 = icmp slt i64 %r78, 508		
	br i1 %r80, label %"file loop.c, line 10, in inner loop at depth 0, bb65", label %"file loop.c, line 16, bb39"

"file loop.c, line 16, bb39":		
	%r84 = add i64 %"$vals_WR0_S15.0", 409927646082434475		
	%r885 = getelementptr [512 x i64]* %vals, i64 0, i64 509		
	store i64 %r84, i64* %r885, align 8
	%r90 = add i64 %"$vals_WR0_S15.0", 491913175298921370		
	%r944 = getelementptr [512 x i64]* %vals, i64 0, i64 510		
	store i64 %r90, i64* %r944, align 16
	%r1003 = getelementptr [512 x i64]* %vals, i64 0, i64 511		
	%r96.c = add i64 %"$vals_WR0_S15.0", 573898704515408265		
	store i64 %r96.c, i64* %r1003, align 8
	br label %"file loop.c, line 10, in inner loop at depth 0, bb13"

"file loop.c, line 10, in inner loop at depth 0, bb13":		
	%"$i_S17.0" = phi i64 [ 0, %"file loop.c, line 16, bb39" ], [ %r111, %"file loop.c, line 19, in inner loop at depth 0, bb25" ]		
	br label %"file loop.c, line 19, in inner loop at depth 0, bb6"

"file loop.c, line 19, in inner loop at depth 0, bb6":		
	%r102 = add i64 %"$i_S17.0", 1		
	%r106 = add i64 %"$i_S17.0", 1		
	%r1071 = getelementptr [512 x i64]* %vals, i64 0, i64 %r106		
	%r1072 = load i64* %r1071, align 8		
	%r109 = call i32 (...)* @printf([17 x i8]* bitcast ([3 x i64]* @"initialized$$$CFE_id_3d9e38d5_main" to [17 x i8]*), i64 %r102, i64 %r1072)		
	br label %"file loop.c, line 19, in inner loop at depth 0, bb25"

"file loop.c, line 19, in inner loop at depth 0, bb25":		
	%r111 = add i64 %"$i_S17.0", 1		
	%r113 = icmp slt i64 %r111, 511		
	br i1 %r113, label %"file loop.c, line 10, in inner loop at depth 0, bb13", label %"file loop.c, line 19, bb10"

"file loop.c, line 19, bb10":
	ret i32 0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

declare i32 @printf(...)

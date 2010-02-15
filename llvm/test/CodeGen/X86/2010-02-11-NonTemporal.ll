; RUN: llc < %s -march=x86-64 | FileCheck %s
; CHECK: movnt
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@blk_ = linkonce global [700000 x i64] zeroinitializer, align 64		; <[700000 x i64]*> [#uses=3]
@_dm_my_pe = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]
@_dm_pes_in_prog = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]
@_dm_npes_div_mult = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]
@_dm_npes_div_shift = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]
@_dm_pe_addr_loc = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]
@_dm_offset_addr_mask = external global [1 x i64], align 64		; <[1 x i64]*> [#uses=1]

!0 = metadata !{ i32 1 }

define void @sub_(i32* noalias %n) {
"file movnt.f90, line 2, bb1":	; srcLine 2
	%n1 = alloca i32*, align 8		; <i32**> [#uses=1]	; [oox.12 : sln.2]
	%i = alloca i32, align 4		; <i32*> [#uses=0]	; [oox.14 : sln.2]
	%"$LIS_E0" = alloca i64, align 8		; <i64*> [#uses=2]	; [oox.72 : sln.2]
	%"$LCS_0" = alloca i64, align 8		; <i64*> [#uses=7]	; [oox.73 : sln.2]
	%"$SI_S1" = alloca i64, align 8		; <i64*> [#uses=4]	; [oox.75 : sln.2]
	%"$LCS_S2" = alloca <2 x double>, align 16		; <<2 x double>*> [#uses=6]	; [oox.76 : sln.2]
	%"$LC_S3" = alloca i64, align 8		; <i64*> [#uses=4]	; [oox.77 : sln.2]
	volatile store i32* %n, i32** %n1	; [oox.12 : sln.2]
	br label %"file movnt.f90, line 2, bb32"	; [oox.0 : sln.0]

"file movnt.f90, line 2, bb32":	; srcLine 2		; preds = %"file movnt.f90, line 2, bb1"
	store i64 -50000, i64* %"$LC_S3", align 8	; [oox.160 : sln.10]
	store i64 0, i64* %"$SI_S1", align 8	; [oox.161 : sln.10]
	store i64 ptrtoint ([700000 x i64]* @blk_ to i64), i64* %"$LIS_E0", align 8	; [oox.162 : sln.11]
	br label %"file movnt.f90, line 2, in inner vector loop at depth 0, bb25"	; [oox.0 : sln.0]

"file movnt.f90, line 2, in inner vector loop at depth 0, bb25":	; srcLine 2		; preds = %"file movnt.f90, line 11, in inner vector loop at depth 0, bb23", %"file movnt.f90, line 2, bb32"
	br label %"file movnt.f90, line 11, in inner vector loop at depth 0, bb23"	; [oox.0 : sln.0]

"file movnt.f90, line 11, in inner vector loop at depth 0, bb23":	; srcLine 11		; preds = %"file movnt.f90, line 2, in inner vector loop at depth 0, bb25"
	%r = load i64* %"$LIS_E0", align 8		; <i64> [#uses=1]	; [oox.159 : sln.11]
	%r2 = load i64* %"$SI_S1", align 8		; <i64> [#uses=1]	; [oox.159 : sln.11]
	%r3 = add i64 %r, %r2		; <i64> [#uses=1]	; [oox.159 : sln.11]
	store i64 %r3, i64* %"$LCS_0", align 8	; [oox.159 : sln.11]
	%r4 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.160 : sln.11]
	%r5 = add i64 4000000, %r4		; <i64> [#uses=1]	; [oox.160 : sln.11]
	%r6 = inttoptr i64 %r5 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.160 : sln.11]
	%r8 = load <2 x double>* %r6, align 16, !nontemporal !0		; <<2 x double>> [#uses=1]	; [oox.160 : sln.11]
	store <2 x double> %r8, <2 x double>* %"$LCS_S2", align 8	; [oox.160 : sln.11]
	%r9 = load <2 x double>* %"$LCS_S2", align 8		; <<2 x double>> [#uses=1]	; [oox.161 : sln.11]
	%r10 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.161 : sln.11]
	%r11 = inttoptr i64 %r10 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.161 : sln.11]
	store <2 x double> %r9, <2 x double>* %r11, align 16, !nontemporal !0	; [oox.161 : sln.11]
	%r13 = load <2 x double>* %"$LCS_S2", align 8		; <<2 x double>> [#uses=1]	; [oox.162 : sln.12]
	%r14 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.162 : sln.12]
	%r15 = add i64 800000, %r14		; <i64> [#uses=1]	; [oox.162 : sln.12]
	%r16 = inttoptr i64 %r15 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.162 : sln.12]
	store <2 x double> %r13, <2 x double>* %r16, align 16, !nontemporal !0	; [oox.162 : sln.12]
	%r18 = load <2 x double>* %"$LCS_S2", align 8		; <<2 x double>> [#uses=1]	; [oox.163 : sln.13]
	%r19 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.163 : sln.13]
	%r20 = add i64 1600000, %r19		; <i64> [#uses=1]	; [oox.163 : sln.13]
	%r21 = inttoptr i64 %r20 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.163 : sln.13]
	store <2 x double> %r18, <2 x double>* %r21, align 16, !nontemporal !0	; [oox.163 : sln.13]
	%r23 = load <2 x double>* %"$LCS_S2", align 8		; <<2 x double>> [#uses=1]	; [oox.164 : sln.14]
	%r24 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.164 : sln.14]
	%r25 = add i64 2400000, %r24		; <i64> [#uses=1]	; [oox.164 : sln.14]
	%r26 = inttoptr i64 %r25 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.164 : sln.14]
	store <2 x double> %r23, <2 x double>* %r26, align 16, !nontemporal !0	; [oox.164 : sln.14]
	%r28 = load <2 x double>* %"$LCS_S2", align 8		; <<2 x double>> [#uses=1]	; [oox.165 : sln.15]
	%r29 = load i64* %"$LCS_0", align 8		; <i64> [#uses=1]	; [oox.165 : sln.15]
	%r30 = add i64 3200000, %r29		; <i64> [#uses=1]	; [oox.165 : sln.15]
	%r31 = inttoptr i64 %r30 to <2 x double>*		; <<2 x double>*> [#uses=2]	; [oox.165 : sln.15]
	store <2 x double> %r28, <2 x double>* %r31, align 16, !nontemporal !0	; [oox.165 : sln.15]
	%r33 = load i64* %"$SI_S1", align 8		; <i64> [#uses=1]	; [oox.166 : sln.10]
	%r34 = add i64 16, %r33		; <i64> [#uses=1]	; [oox.166 : sln.10]
	store i64 %r34, i64* %"$SI_S1", align 8	; [oox.166 : sln.10]
	%r35 = load i64* %"$LC_S3", align 8		; <i64> [#uses=1]	; [oox.167 : sln.10]
	%r36 = add i64 1, %r35		; <i64> [#uses=1]	; [oox.167 : sln.10]
	store i64 %r36, i64* %"$LC_S3", align 8	; [oox.167 : sln.10]
	%r37 = load i64* %"$LC_S3", align 8		; <i64> [#uses=1]	; [oox.168 : sln.10]
	%r38 = icmp slt i64 %r37, 0		; <i1> [#uses=1]	; [oox.168 : sln.10]
	%r39 = zext i1 %r38 to i64		; <i64> [#uses=1]	; [oox.168 : sln.10]
	%r40 = icmp ne i64 %r39, 0		; <i1> [#uses=1]	; [oox.168 : sln.10]
	br i1 %r40, label %"file movnt.f90, line 2, in inner vector loop at depth 0, bb25", label %"file movnt.f90, line 18, bb5"	; [oox.168 : sln.10]

"file movnt.f90, line 18, bb5":	; srcLine 18		; preds = %"file movnt.f90, line 11, in inner vector loop at depth 0, bb23"
	ret void	; [oox.159 : sln.18]
}

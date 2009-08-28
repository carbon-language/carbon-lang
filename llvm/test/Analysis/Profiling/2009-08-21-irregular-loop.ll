; RUN: llvm-as < %s | opt -insert-edge-profiling > %t1
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1 1
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1 1 2
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1 1 2 3
; RUN: lli -load %llvmlibsdir/profile_rt%shlibext %t1 1 2 3 4
; RUN: mv llvmprof.out %t2
; RUN: llvm-prof -print-all-code %t1 %t2 | tee %t3 | FileCheck %s
; CHECK:  1.     5/5 main
; CHECK:  1. 19.5745%    46/235	main() - bb6
; CHECK:  2. 15.7447%    37/235	main() - bb2
; CHECK:  3. 15.3191%    36/235	main() - bb3
; CHECK:  4. 15.3191%    36/235	main() - bb5
; CHECK:  5. 7.23404%    17/235	main() - bb10
; CHECK:  6. 5.53191%    13/235	main() - bb
; CHECK:  7. 5.10638%    12/235	main() - bb9
; CHECK:  8. 4.25532%    10/235	main() - bb1
; CHECK:  9. 3.82979%     9/235	main() - bb7
; CHECK: 10. 2.55319%     6/235	main() - bb11
; CHECK: 11. 2.12766%     5/235	main() - entry
; CHECK: 12. 1.70213%     4/235	main() - bb10.bb11_crit_edge
; CHECK: 13. 1.2766%     3/235	main() - bb8
; CHECK: 14. 0.425532%     1/235	main() - bb4
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@.str = internal constant [6 x i8] c"franz\00"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [9 x i8] c"argc > 2\00"		; <[9 x i8]*> [#uses=1]
@.str2 = internal constant [9 x i8] c"argc = 1\00"		; <[9 x i8]*> [#uses=1]
@.str3 = internal constant [6 x i8] c"fritz\00"		; <[6 x i8]*> [#uses=1]
@.str4 = internal constant [10 x i8] c"argc <= 1\00"		; <[10 x i8]*> [#uses=1]

; CHECK:;;; %main called 5 times.
; CHECK:;;;
define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
; CHECK:entry:
; CHECK:	;;; Basic block executed 5 times.
	%argc_addr = alloca i32		; <i32*> [#uses=4]
	%argv_addr = alloca i8**		; <i8***> [#uses=1]
	%retval = alloca i32		; <i32*> [#uses=2]
	%j = alloca i32		; <i32*> [#uses=4]
	%i = alloca i32		; <i32*> [#uses=4]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %argc, i32* %argc_addr
	store i8** %argv, i8*** %argv_addr
	store i32 0, i32* %i, align 4
	br label %bb10
; CHECK:	;;; Out-edge counts: [5.000000e+00 -> bb10]

bb:		; preds = %bb10
; CHECK:bb:
; CHECK:	;;; Basic block executed 13 times.
	%1 = load i32* %argc_addr, align 4		; <i32> [#uses=1]
	%2 = icmp sgt i32 %1, 1		; <i1> [#uses=1]
	br i1 %2, label %bb1, label %bb8
; CHECK:	;;; Out-edge counts: [1.000000e+01 -> bb1] [3.000000e+00 -> bb8]

bb1:		; preds = %bb
; CHECK:bb1:
; CHECK:	;;; Basic block executed 10 times.
	store i32 0, i32* %j, align 4
	br label %bb6
; CHECK:	;;; Out-edge counts: [1.000000e+01 -> bb6]

bb2:		; preds = %bb6
; CHECK:bb2:
; CHECK:	;;; Basic block executed 37 times.
	%3 = call i32 @puts(i8* getelementptr ([6 x i8]* @.str, i64 0, i64 0)) nounwind		; <i32> [#uses=0]
	%4 = load i32* %argc_addr, align 4		; <i32> [#uses=1]
	%5 = icmp sgt i32 %4, 2		; <i1> [#uses=1]
	br i1 %5, label %bb3, label %bb4
; CHECK:	;;; Out-edge counts: [3.600000e+01 -> bb3] [1.000000e+00 -> bb4]

bb3:		; preds = %bb2
; CHECK:bb3:
; CHECK:	;;; Basic block executed 36 times.
	%6 = call i32 @puts(i8* getelementptr ([9 x i8]* @.str1, i64 0, i64 0)) nounwind		; <i32> [#uses=0]
	br label %bb5
; CHECK:	;;; Out-edge counts: [3.600000e+01 -> bb5]

bb4:		; preds = %bb2
; CHECK:bb4:
; CHECK:	;;; Basic block executed 1 times.
	%7 = call i32 @puts(i8* getelementptr ([9 x i8]* @.str2, i64 0, i64 0)) nounwind		; <i32> [#uses=0]
	br label %bb11
; CHECK:	;;; Out-edge counts: [1.000000e+00 -> bb11]

bb5:		; preds = %bb3
; CHECK:bb5:
; CHECK:	;;; Basic block executed 36 times.
	%8 = call i32 @puts(i8* getelementptr ([6 x i8]* @.str3, i64 0, i64 0)) nounwind		; <i32> [#uses=0]
	%9 = load i32* %j, align 4		; <i32> [#uses=1]
	%10 = add i32 %9, 1		; <i32> [#uses=1]
	store i32 %10, i32* %j, align 4
	br label %bb6
; CHECK:	;;; Out-edge counts: [3.600000e+01 -> bb6]

bb6:		; preds = %bb5, %bb1
; CHECK:bb6:
; CHECK:	;;; Basic block executed 46 times.
	%11 = load i32* %j, align 4		; <i32> [#uses=1]
	%12 = load i32* %argc_addr, align 4		; <i32> [#uses=1]
	%13 = icmp slt i32 %11, %12		; <i1> [#uses=1]
	br i1 %13, label %bb2, label %bb7
; CHECK:	;;; Out-edge counts: [3.700000e+01 -> bb2] [9.000000e+00 -> bb7]

bb7:		; preds = %bb6
; CHECK:bb7:
; CHECK:	;;; Basic block executed 9 times.
	br label %bb9
; CHECK:	;;; Out-edge counts: [9.000000e+00 -> bb9]

bb8:		; preds = %bb
; CHECK:bb8:
; CHECK:	;;; Basic block executed 3 times.
	%14 = call i32 @puts(i8* getelementptr ([10 x i8]* @.str4, i64 0, i64 0)) nounwind		; <i32> [#uses=0]
	br label %bb9
; CHECK:	;;; Out-edge counts: [3.000000e+00 -> bb9]

bb9:		; preds = %bb8, %bb7
; CHECK:bb9:
; CHECK:	;;; Basic block executed 12 times.
	%15 = load i32* %i, align 4		; <i32> [#uses=1]
	%16 = add i32 %15, 1		; <i32> [#uses=1]
	store i32 %16, i32* %i, align 4
	br label %bb10
; CHECK:	;;; Out-edge counts: [1.200000e+01 -> bb10]

bb10:		; preds = %bb9, %entry
; CHECK:	;;; Basic block executed 17 times.
	%17 = load i32* %i, align 4		; <i32> [#uses=1]
	%18 = icmp ne i32 %17, 3		; <i1> [#uses=1]
	br i1 %18, label %bb, label %bb11
; CHECK:	;;; Out-edge counts: [1.300000e+01 -> bb] [4.000000e+00 -> bb10.bb11_crit_edge]
; CHECK:	;;; Basic block executed 4 times.
; CHECK:	;;; Out-edge counts: [5.000000e+00 -> bb11]

bb11:		; preds = %bb10, %bb4
; CHECK:bb11:
; CHECK:	;;; Basic block executed 6 times.
	store i32 0, i32* %0, align 4
	%19 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %19, i32* %retval, align 4
	br label %return

return:		; preds = %bb11
; CHECK:return:
	%retval12 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval12
}

declare i32 @puts(i8*)

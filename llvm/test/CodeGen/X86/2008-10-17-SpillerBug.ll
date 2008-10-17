; RUN: llvm-as < %s | llc -mtriple=x86_64-pc-linux-gnu -regalloc=pbqp -stats |& grep {Number of dead stores elided} | grep 2
; PR2898

	%struct.BiContextType = type { i16, i8 }
	%struct.Bitstream = type { i32, i32, i32, i32, i8*, i32 }
	%struct.DataPartition = type { %struct.Bitstream*, %struct.DecodingEnvironment, i32 (%struct.SyntaxElement*, %struct.ImageParameters*, %struct.DataPartition*)* }
	%struct.DecRefPicMarking_t = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_t* }
	%struct.DecodingEnvironment = type { i32, i32, i32, i32, i32, i8*, i32* }
	%struct.ImageParameters = type { i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [16 x i16]], [6 x [32 x i32]], [16 x [16 x i32]], [4 x [12 x [4 x [4 x i32]]]], [16 x i32], i8**, i32*, i32***, i32**, i32, i32, i32, i32, %struct.Slice*, %struct.Macroblock*, i32, i32, i32, i32, i32, i32, %struct.DecRefPicMarking_t*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32***, i32***, i32****, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [3 x [2 x i32]], i32, i32, i64, i64, %struct.timeb, %struct.timeb, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.Macroblock = type { i32, [2 x i32], i32, i32, %struct.Macroblock*, %struct.Macroblock*, i32, [2 x [4 x [4 x [2 x i32]]]], i32, i64, i64, i32, i32, [4 x i8], [4 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.MotionInfoContexts = type { [4 x [11 x %struct.BiContextType]], [2 x [9 x %struct.BiContextType]], [2 x [10 x %struct.BiContextType]], [2 x [6 x %struct.BiContextType]], [4 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x %struct.BiContextType] }
	%struct.Slice = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.DataPartition*, %struct.MotionInfoContexts*, %struct.TextureInfoContexts*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (%struct.ImageParameters*, %struct.inp_par*)*, i32, i32, i32, i32 }
	%struct.SyntaxElement = type { i32, i32, i32, i32, i32, i32, i32, i32, void (i32, i32, i32*, i32*)*, void (%struct.SyntaxElement*, %struct.ImageParameters*, %struct.DecodingEnvironment*)* }
	%struct.TextureInfoContexts = type { [2 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x [4 x %struct.BiContextType]], [10 x [4 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]] }
	%struct.inp_par = type { [1000 x i8], [1000 x i8], [1000 x i8], i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.timeb = type { i64, i16, i16, i16 }

define i1 @itrans8x8_bb15_bb15_2E_ce(%struct.ImageParameters* %img, i32 %ioff, [8 x [8 x i32]]* %m6, i64, i64, i64, i64, i64, i64, i64, i64, i32 %i.2.reg2mem.0, i32* %.out, i32* %.out1, i32* %.out2, i32* %.out3, i32* %.out4, i32* %.out5, i32* %.out6, i32* %.out7, i32* %.out8, i32* %indvar.next58.out) {
newFuncRoot:
	br label %bb15.ce

codeRepl1.exitStub:		; preds = %bb15.ce
	store i32 %8, i32* %.out
	store i32 %24, i32* %.out1
	store i32 %25, i32* %.out2
	store i32 %26, i32* %.out3
	store i32 %27, i32* %.out4
	store i32 %53, i32* %.out5
	store i32 %55, i32* %.out6
	store i32 %57, i32* %.out7
	store i32 %59, i32* %.out8
	store i32 %indvar.next58, i32* %indvar.next58.out
	ret i1 true

bb15.bb15_crit_edge.exitStub:		; preds = %bb15.ce
	store i32 %8, i32* %.out
	store i32 %24, i32* %.out1
	store i32 %25, i32* %.out2
	store i32 %26, i32* %.out3
	store i32 %27, i32* %.out4
	store i32 %53, i32* %.out5
	store i32 %55, i32* %.out6
	store i32 %57, i32* %.out7
	store i32 %59, i32* %.out8
	store i32 %indvar.next58, i32* %indvar.next58.out
	ret i1 false

bb15.ce:		; preds = %newFuncRoot
	%8 = add i32 %i.2.reg2mem.0, %ioff		; <i32> [#uses=3]
	%9 = sext i32 %i.2.reg2mem.0 to i64		; <i64> [#uses=8]
	%10 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 0		; <i32*> [#uses=1]
	%11 = load i32* %10, align 4		; <i32> [#uses=2]
	%12 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 4		; <i32*> [#uses=1]
	%13 = load i32* %12, align 4		; <i32> [#uses=2]
	%14 = add i32 %13, %11		; <i32> [#uses=2]
	%15 = sub i32 %11, %13		; <i32> [#uses=2]
	%16 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 2		; <i32*> [#uses=1]
	%17 = load i32* %16, align 4		; <i32> [#uses=2]
	%18 = ashr i32 %17, 1		; <i32> [#uses=1]
	%19 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 6		; <i32*> [#uses=1]
	%20 = load i32* %19, align 4		; <i32> [#uses=2]
	%21 = sub i32 %18, %20		; <i32> [#uses=2]
	%22 = ashr i32 %20, 1		; <i32> [#uses=1]
	%23 = add i32 %22, %17		; <i32> [#uses=2]
	%24 = add i32 %23, %14		; <i32> [#uses=4]
	%25 = add i32 %21, %15		; <i32> [#uses=4]
	%26 = sub i32 %15, %21		; <i32> [#uses=4]
	%27 = sub i32 %14, %23		; <i32> [#uses=4]
	%28 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 5		; <i32*> [#uses=1]
	%29 = load i32* %28, align 4		; <i32> [#uses=4]
	%30 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 3		; <i32*> [#uses=1]
	%31 = load i32* %30, align 4		; <i32> [#uses=4]
	%32 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 7		; <i32*> [#uses=1]
	%33 = load i32* %32, align 4		; <i32> [#uses=4]
	%34 = ashr i32 %33, 1		; <i32> [#uses=1]
	%35 = sub i32 %29, %31		; <i32> [#uses=1]
	%36 = sub i32 %35, %33		; <i32> [#uses=1]
	%37 = sub i32 %36, %34		; <i32> [#uses=2]
	%38 = getelementptr [8 x [8 x i32]]* %m6, i64 0, i64 %9, i64 1		; <i32*> [#uses=1]
	%39 = load i32* %38, align 4		; <i32> [#uses=4]
	%40 = ashr i32 %31, 1		; <i32> [#uses=1]
	%41 = add i32 %33, %39		; <i32> [#uses=1]
	%42 = sub i32 %41, %31		; <i32> [#uses=1]
	%43 = sub i32 %42, %40		; <i32> [#uses=2]
	%44 = ashr i32 %29, 1		; <i32> [#uses=1]
	%45 = sub i32 %33, %39		; <i32> [#uses=1]
	%46 = add i32 %45, %29		; <i32> [#uses=1]
	%47 = add i32 %46, %44		; <i32> [#uses=2]
	%48 = ashr i32 %39, 1		; <i32> [#uses=1]
	%49 = add i32 %29, %31		; <i32> [#uses=1]
	%50 = add i32 %49, %39		; <i32> [#uses=1]
	%51 = add i32 %50, %48		; <i32> [#uses=2]
	%52 = ashr i32 %51, 2		; <i32> [#uses=1]
	%53 = add i32 %52, %37		; <i32> [#uses=4]
	%54 = ashr i32 %37, 2		; <i32> [#uses=1]
	%55 = sub i32 %51, %54		; <i32> [#uses=4]
	%56 = ashr i32 %47, 2		; <i32> [#uses=1]
	%57 = add i32 %56, %43		; <i32> [#uses=4]
	%58 = ashr i32 %43, 2		; <i32> [#uses=1]
	%59 = sub i32 %58, %47		; <i32> [#uses=4]
	%60 = add i32 %55, %24		; <i32> [#uses=1]
	%61 = sext i32 %8 to i64		; <i64> [#uses=8]
	%62 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %0, i64 %61		; <i32*> [#uses=1]
	store i32 %60, i32* %62, align 4
	%63 = add i32 %59, %25		; <i32> [#uses=1]
	%64 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %1, i64 %61		; <i32*> [#uses=1]
	store i32 %63, i32* %64, align 4
	%65 = add i32 %57, %26		; <i32> [#uses=1]
	%66 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %2, i64 %61		; <i32*> [#uses=1]
	store i32 %65, i32* %66, align 4
	%67 = add i32 %53, %27		; <i32> [#uses=1]
	%68 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %3, i64 %61		; <i32*> [#uses=1]
	store i32 %67, i32* %68, align 4
	%69 = sub i32 %27, %53		; <i32> [#uses=1]
	%70 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %4, i64 %61		; <i32*> [#uses=1]
	store i32 %69, i32* %70, align 4
	%71 = sub i32 %26, %57		; <i32> [#uses=1]
	%72 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %5, i64 %61		; <i32*> [#uses=1]
	store i32 %71, i32* %72, align 4
	%73 = sub i32 %25, %59		; <i32> [#uses=1]
	%74 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %6, i64 %61		; <i32*> [#uses=1]
	store i32 %73, i32* %74, align 4
	%75 = sub i32 %24, %55		; <i32> [#uses=1]
	%76 = getelementptr %struct.ImageParameters* %img, i64 0, i32 27, i64 %7, i64 %61		; <i32*> [#uses=1]
	store i32 %75, i32* %76, align 4
	%indvar.next58 = add i32 %i.2.reg2mem.0, 1		; <i32> [#uses=3]
	%exitcond59 = icmp eq i32 %indvar.next58, 8		; <i1> [#uses=1]
	br i1 %exitcond59, label %codeRepl1.exitStub, label %bb15.bb15_crit_edge.exitStub
}

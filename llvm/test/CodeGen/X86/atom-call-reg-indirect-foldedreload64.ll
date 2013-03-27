; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=atom 2>&1 | \
; RUN:     grep "callq" | not grep "("
; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=core2 2>&1 | \
; RUN:     grep "callq" | grep "8-byte Folded Reload"

%struct.targettype = type {i64}
%struct.op_ptr1 = type opaque
%struct.op_ptr2 = type opaque
%union.anon = type { [8 x i32], [48 x i8] }
%struct.const1 = type { [64 x i16], i8 }
%struct.const2 = type { [17 x i8], [256 x i8], i8 }
%struct.coef1 = type { %struct.ref1, i32, i32, i32, [10 x [64 x i16]*] }

%struct.ref1 = type { void (%struct.ref2*)*, i32 (%struct.ref2*)*, void (%struct.ref2*)*, i32 (%struct.ref2*, i8***)*, %struct.op_ptr2** }
%struct.ref2 = type { %struct.localref13*, %struct.localref15*, %struct.localref12*, i8*, i8, i32, %struct.localref11*, i32, i32, i32, i32, i32, i32, i32, double, i8, i8, i32, i8, i8, i8, i32, i8, i32, i8, i8, i8, i32, i32, i32, i32, i32, i32, i8**, i32, i32, i32, i32, i32, [64 x i32]*, [4 x %struct.const1*], [4 x %struct.const2*], [4 x %struct.const2*], i32, %struct.ref3*, i8, i8, [16 x i8], [16 x i8], [16 x i8], i32, i8, i8, i8, i8, i16, i16, i8, i8, i8, %struct.localref10*, i32, i32, i32, i32, i8*, i32, [4 x %struct.ref3*], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, %struct.localref8*, %struct.localref9*, %struct.ref1*, %struct.localref7*, %struct.localref6*, %struct.localref5*, %struct.localref1*, %struct.ref4*, %struct.localref2*, %struct.localref3*, %struct.localref4* }
%struct.ref3 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, %struct.const1*, i8* }
%struct.ref4 = type { void (%struct.ref2*)*, [5 x void (%struct.ref2*, %struct.ref3*, i16*, i8**, i32)*] }

%struct.localref1 = type { void (%struct.ref2*)*, i8 (%struct.ref2*, [64 x i16]**)*, i8 }
%struct.localref2 = type { void (%struct.ref2*)*, void (%struct.ref2*, i8***, i32*, i32, i8**, i32*, i32)*, i8 }
%struct.localref3 = type { void (%struct.ref2*)*, void (%struct.ref2*, i8***, i32, i8**, i32)* }
%struct.localref4 = type { {}*, void (%struct.ref2*, i8**, i8**, i32)*, void (%struct.ref2*)*, void (%struct.ref2*)* }
%struct.localref5 = type { void (%struct.ref2*)*, i32 (%struct.ref2*)*, i8 (%struct.ref2*)*, i8, i8, i32, i32 }
%struct.localref6 = type { i32 (%struct.ref2*)*, void (%struct.ref2*)*, void (%struct.ref2*)*, void (%struct.ref2*)*, i8, i8 }
%struct.localref7 = type { void (%struct.ref2*, i32)*, void (%struct.ref2*, i8***, i32*, i32, i8**, i32*, i32)* }
%struct.localref8 = type { void (%struct.ref2*)*, void (%struct.ref2*)*, i8 }
%struct.localref9 = type { void (%struct.ref2*, i32)*, void (%struct.ref2*, i8**, i32*, i32)* }
%struct.localref10 = type { %struct.localref10*, i8, i32, i32, i8* }
%struct.localref11 = type { i8*, %struct.targettype, void (%struct.ref2*)*, i8 (%struct.ref2*)*, void (%struct.ref2*, %struct.targettype)*, i8 (%struct.ref2*, i32)*, void (%struct.ref2*)* }
%struct.localref12 = type { {}*, %struct.targettype, %struct.targettype, i32, i32 }
%struct.localref13 = type { void (%struct.localref14*)*, void (%struct.localref14*, i32)*, void (%struct.localref14*)*, void (%struct.localref14*, i8*)*, void (%struct.localref14*)*, i32, %union.anon, i32, %struct.targettype, i8**, i32, i8**, i32, i32 }
%struct.localref14 = type { %struct.localref13*, %struct.localref15*, %struct.localref12*, i8*, i8, i32 }
%struct.localref15 = type { i8* (%struct.localref14*, i32, %struct.targettype)*, i8* (%struct.localref14*, i32, %struct.targettype)*, i8** (%struct.localref14*, i32, i32, i32)*, [64 x i16]** (%struct.localref14*, i32, i32, i32)*, %struct.op_ptr1* (%struct.localref14*, i32, i8, i32, i32, i32)*, %struct.op_ptr2* (%struct.localref14*, i32, i8, i32, i32, i32)*, {}*, i8** (%struct.localref14*, %struct.op_ptr1*, i32, i32, i8)*, [64 x i16]** (%struct.localref14*, %struct.op_ptr2*, i32, i32, i8)*, void (%struct.localref14*, i32)*, {}*, %struct.targettype, %struct.targettype}

define internal i32 @foldedreload(%struct.ref2* %cinfo, i8*** nocapture %output1) {
  %1 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 79
  %2 = load %struct.ref1** %1, align 8
  %3 = bitcast %struct.ref1* %2 to %struct.coef1*
  %4 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 68
  %5 = load i32* %4, align 4
  %6 = add i32 %5, -1
  %7 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 64
  %8 = load i32* %7, align 4
  %9 = add i32 %8, -1
  %10 = getelementptr inbounds %struct.coef1* %3, i64 0, i32 2
  %11 = load i32* %10, align 4
  %12 = getelementptr inbounds %struct.ref1* %2, i64 1, i32 1
  %13 = bitcast i32 (%struct.ref2*)** %12 to i32*
  %14 = load i32* %13, align 4
  %15 = icmp slt i32 %11, %14
  br i1 %15, label %.lr.ph18, label %._crit_edge19

.lr.ph18:
  %16 = getelementptr inbounds %struct.ref1* %2, i64 1
  %17 = bitcast %struct.ref1* %16 to i32*
  %18 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 66
  %19 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 84
  %20 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 36
  %21 = getelementptr inbounds %struct.ref1* %2, i64 1, i32 2
  %22 = bitcast void (%struct.ref2*)** %21 to [10 x [64 x i16]*]*
  %.pre = load i32* %17, align 4
  br label %23

; <label>:23
  %24 = phi i32 [ %14, %.lr.ph18 ], [ %92, %91 ]
  %25 = phi i32 [ %.pre, %.lr.ph18 ], [ 0, %91 ]
  %var1.015 = phi i32 [ %11, %.lr.ph18 ], [ %93, %91 ]
  %26 = icmp ugt i32 %25, %6
  br i1 %26, label %91, label %.preheader7.lr.ph

.preheader7.lr.ph:
  %.pre24 = load i32* %18, align 4
  br label %.preheader7

.preheader7:
  %27 = phi i32 [ %.pre24, %.preheader7.lr.ph ], [ %88, %._crit_edge11 ]
  %var2.012 = phi i32 [ %25, %.preheader7.lr.ph ], [ %89, %._crit_edge11 ]
  %28 = icmp sgt i32 %27, 0
  br i1 %28, label %.lr.ph10, label %._crit_edge11

.lr.ph10:
  %29 = phi i32 [ %27, %.preheader7 ], [ %85, %84 ]
  %indvars.iv21 = phi i64 [ 0, %.preheader7 ], [ %indvars.iv.next22, %84 ]
  %var4.09 = phi i32 [ 0, %.preheader7 ], [ %var4.1.lcssa, %84 ]
  %30 = getelementptr inbounds %struct.ref2* %cinfo, i64 0, i32 67, i64 %indvars.iv21
  %31 = load %struct.ref3** %30, align 8
  %32 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 1
  %33 = load i32* %32, align 4
  %34 = sext i32 %33 to i64
  %35 = load %struct.ref4** %19, align 8
  %36 = getelementptr inbounds %struct.ref4* %35, i64 0, i32 1, i64 %34
  %37 = load void (%struct.ref2*, %struct.ref3*, i16*, i8**, i32)** %36, align 8
  %38 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 17
  %39 = load i32* %38, align 4
  %40 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 9
  %41 = load i32* %40, align 4
  %42 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 16
  %43 = load i32* %42, align 4
  %44 = mul i32 %43, %var2.012
  %45 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 14
  %46 = load i32* %45, align 4
  %47 = icmp sgt i32 %46, 0
  br i1 %47, label %.lr.ph6, label %84

.lr.ph6:
  %48 = mul nsw i32 %41, %var1.015
  %49 = getelementptr inbounds i8*** %output1, i64 %34
  %50 = sext i32 %48 to i64
  %51 = load i8*** %49, align 8
  %52 = getelementptr inbounds i8** %51, i64 %50
  %53 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 13
  %54 = getelementptr inbounds %struct.ref3* %31, i64 0, i32 18
  %55 = icmp sgt i32 %39, 0
  br i1 %55, label %.lr.ph6.split.us, label %.lr.ph6..lr.ph6.split_crit_edge

.lr.ph6..lr.ph6.split_crit_edge:
  br label %._crit_edge28

.lr.ph6.split.us:
  %56 = phi i32 [ %64, %._crit_edge30 ], [ %46, %.lr.ph6 ]
  %57 = phi i32 [ %65, %._crit_edge30 ], [ %41, %.lr.ph6 ]
  %var4.15.us = phi i32 [ %67, %._crit_edge30 ], [ %var4.09, %.lr.ph6 ]
  %output2.04.us = phi i8** [ %71, %._crit_edge30 ], [ %52, %.lr.ph6 ]
  %var5.03.us = phi i32 [ %68, %._crit_edge30 ], [ 0, %.lr.ph6 ]
  %58 = load i32* %20, align 4
  %59 = icmp ult i32 %58, %9
  br i1 %59, label %.lr.ph.us, label %60

; <label>:60
  %61 = add nsw i32 %var5.03.us, %var1.015
  %62 = load i32* %54, align 4
  %63 = icmp slt i32 %61, %62
  br i1 %63, label %.lr.ph.us, label %._crit_edge29

._crit_edge29:
  %64 = phi i32 [ %.pre25.pre, %.loopexit.us ], [ %56, %60 ]
  %65 = phi i32 [ %77, %.loopexit.us ], [ %57, %60 ]
  %66 = load i32* %53, align 4
  %67 = add nsw i32 %66, %var4.15.us
  %68 = add nsw i32 %var5.03.us, 1
  %69 = icmp slt i32 %68, %64
  br i1 %69, label %._crit_edge30, label %._crit_edge

._crit_edge30:
  %70 = sext i32 %65 to i64
  %71 = getelementptr inbounds i8** %output2.04.us, i64 %70
  br label %.lr.ph6.split.us

; <label>:72
  %indvars.iv = phi i64 [ 0, %.lr.ph.us ], [ %indvars.iv.next, %72 ]
  %var3.02.us = phi i32 [ %44, %.lr.ph.us ], [ %78, %72 ]
  %73 = add nsw i64 %indvars.iv, %79
  %74 = getelementptr inbounds [10 x [64 x i16]*]* %22, i64 0, i64 %73
  %75 = load [64 x i16]** %74, align 8
  %76 = getelementptr inbounds [64 x i16]* %75, i64 0, i64 0
  tail call void %37(%struct.ref2* %cinfo, %struct.ref3* %31, i16* %76, i8** %output2.04.us, i32 %var3.02.us) nounwind
  %77 = load i32* %40, align 4
  %78 = add i32 %77, %var3.02.us
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %39
  br i1 %exitcond, label %.loopexit.us, label %72

.loopexit.us:
  %.pre25.pre = load i32* %45, align 4
  br label %._crit_edge29

.lr.ph.us:
  %79 = sext i32 %var4.15.us to i64
  br label %72

._crit_edge28:
  %var4.15 = phi i32 [ %var4.09, %.lr.ph6..lr.ph6.split_crit_edge ], [ %81, %._crit_edge28 ]
  %var5.03 = phi i32 [ 0, %.lr.ph6..lr.ph6.split_crit_edge ], [ %82, %._crit_edge28 ]
  %80 = load i32* %53, align 4
  %81 = add nsw i32 %80, %var4.15
  %82 = add nsw i32 %var5.03, 1
  %83 = icmp slt i32 %82, %46
  br i1 %83, label %._crit_edge28, label %._crit_edge

._crit_edge:
  %split = phi i32 [ %67, %._crit_edge29 ], [ %81, %._crit_edge28 ]
  %.pre27 = load i32* %18, align 4
  br label %84

; <label>:84
  %85 = phi i32 [ %.pre27, %._crit_edge ], [ %29, %.lr.ph10 ]
  %var4.1.lcssa = phi i32 [ %split, %._crit_edge ], [ %var4.09, %.lr.ph10 ]
  %indvars.iv.next22 = add i64 %indvars.iv21, 1
  %86 = trunc i64 %indvars.iv.next22 to i32
  %87 = icmp slt i32 %86, %85
  br i1 %87, label %.lr.ph10, label %._crit_edge11

._crit_edge11:
  %88 = phi i32 [ %27, %.preheader7 ], [ %85, %84 ]
  %89 = add i32 %var2.012, 1
  %90 = icmp ugt i32 %89, %6
  br i1 %90, label %._crit_edge14, label %.preheader7

._crit_edge14:
  %.pre23 = load i32* %13, align 4
  br label %91

; <label>:91
  %92 = phi i32 [ %.pre23, %._crit_edge14 ], [ %24, %23 ]
  store i32 0, i32* %17, align 4
  %93 = add nsw i32 %var1.015, 1
  %94 = icmp slt i32 %93, %92
  br i1 %94, label %23, label %._crit_edge19

._crit_edge19:
  ret i32 3
}

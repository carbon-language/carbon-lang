; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=atom 2>&1 | \
; RUN:     grep "calll" | not grep "("
; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=core2 2>&1 | \
; RUN:     grep "calll" | grep "4-byte Folded Reload"

%struct.targettype = type {i32}
%struct.op_ptr1 = type opaque
%struct.op_ptr2 = type opaque
%union.anon = type { [8 x i32], [48 x i8] }
%struct.const1 = type { [64 x i16], i8 }
%struct.const2 = type { [17 x i8], [256 x i8], i8 }

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
  %1 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 79
  %2 = load %struct.ref1** %1, align 4
  %3 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 68
  %4 = load i32* %3, align 4
  %5 = add i32 %4, -1
  %6 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 64
  %7 = load i32* %6, align 4
  %8 = add i32 %7, -1
  %9 = getelementptr inbounds %struct.ref1* %2, i32 1, i32 1
  %10 = bitcast i32 (%struct.ref2*)** %9 to i32*
  %11 = load i32* %10, align 4
  %12 = getelementptr inbounds %struct.ref1* %2, i32 1, i32 2
  %13 = bitcast void (%struct.ref2*)** %12 to i32*
  %14 = load i32* %13, align 4
  %15 = icmp slt i32 %11, %14
  br i1 %15, label %.lr.ph18, label %._crit_edge19

.lr.ph18:
  %16 = getelementptr inbounds %struct.ref1* %2, i32 1
  %17 = bitcast %struct.ref1* %16 to i32*
  %18 = getelementptr inbounds %struct.ref1* %16, i32 0, i32 0
  %19 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 66
  %20 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 84
  %21 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 36
  %22 = getelementptr inbounds %struct.ref1* %2, i32 1, i32 3
  %23 = bitcast i32 (%struct.ref2*, i8***)** %22 to [10 x [64 x i16]*]*
  %.pre = load i32* %17, align 4
  br label %24

; <label>:24
  %25 = phi i32 [ %14, %.lr.ph18 ], [ %89, %88 ]
  %26 = phi i32 [ %.pre, %.lr.ph18 ], [ 0, %88 ]
  %var1.015 = phi i32 [ %11, %.lr.ph18 ], [ %90, %88 ]
  %27 = icmp ugt i32 %26, %5
  br i1 %27, label %88, label %.preheader7.lr.ph

.preheader7.lr.ph:
  %.pre24 = load i32* %19, align 4
  br label %.preheader7

.preheader7:
  %28 = phi i32 [ %.pre24, %.preheader7.lr.ph ], [ %85, %._crit_edge11 ]
  %var2.012 = phi i32 [ %26, %.preheader7.lr.ph ], [ %86, %._crit_edge11 ]
  %29 = icmp sgt i32 %28, 0
  br i1 %29, label %.lr.ph10, label %._crit_edge11

.lr.ph10:
  %30 = phi i32 [ %28, %.preheader7 ], [ %82, %81 ]
  %var4.09 = phi i32 [ 0, %.preheader7 ], [ %var4.1.lcssa, %81 ]
  %ci.08 = phi i32 [ 0, %.preheader7 ], [ %83, %81 ]
  %31 = getelementptr inbounds %struct.ref2* %cinfo, i32 0, i32 67, i32 %ci.08
  %32 = load %struct.ref3** %31, align 4
  %33 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 1
  %34 = load i32* %33, align 4
  %35 = load %struct.ref4** %20, align 4
  %36 = getelementptr inbounds %struct.ref4* %35, i32 0, i32 1, i32 %34
  %37 = load void (%struct.ref2*, %struct.ref3*, i16*, i8**, i32)** %36, align 4
  %38 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 17
  %39 = load i32* %38, align 4
  %40 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 9
  %41 = load i32* %40, align 4
  %42 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 16
  %43 = load i32* %42, align 4
  %44 = mul i32 %43, %var2.012
  %45 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 14
  %46 = load i32* %45, align 4
  %47 = icmp sgt i32 %46, 0
  br i1 %47, label %.lr.ph6, label %81

.lr.ph6:
  %48 = getelementptr inbounds i8*** %output1, i32 %34
  %49 = mul nsw i32 %41, %var1.015
  %50 = load i8*** %48, align 4
  %51 = getelementptr inbounds i8** %50, i32 %49
  %52 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 13
  %53 = getelementptr inbounds %struct.ref3* %32, i32 0, i32 18
  %54 = icmp sgt i32 %39, 0
  br i1 %54, label %.lr.ph6.split.us, label %.lr.ph6..lr.ph6.split_crit_edge

.lr.ph6..lr.ph6.split_crit_edge:
  br label %._crit_edge26

.lr.ph6.split.us:
  %55 = phi i32 [ %63, %._crit_edge28 ], [ %46, %.lr.ph6 ]
  %56 = phi i32 [ %64, %._crit_edge28 ], [ %41, %.lr.ph6 ]
  %var4.15.us = phi i32 [ %66, %._crit_edge28 ], [ %var4.09, %.lr.ph6 ]
  %output2.04.us = phi i8** [ %69, %._crit_edge28 ], [ %51, %.lr.ph6 ]
  %var5.03.us = phi i32 [ %67, %._crit_edge28 ], [ 0, %.lr.ph6 ]
  %57 = load i32* %21, align 4
  %58 = icmp ult i32 %57, %8
  br i1 %58, label %.lr.ph.us, label %59

; <label>:59
  %60 = add nsw i32 %var5.03.us, %var1.015
  %61 = load i32* %53, align 4
  %62 = icmp slt i32 %60, %61
  br i1 %62, label %.lr.ph.us, label %._crit_edge27

._crit_edge27:
  %63 = phi i32 [ %.pre23.pre, %.loopexit.us ], [ %55, %59 ]
  %64 = phi i32 [ %74, %.loopexit.us ], [ %56, %59 ]
  %65 = load i32* %52, align 4
  %66 = add nsw i32 %65, %var4.15.us
  %67 = add nsw i32 %var5.03.us, 1
  %68 = icmp slt i32 %67, %63
  br i1 %68, label %._crit_edge28, label %._crit_edge

._crit_edge28:
  %69 = getelementptr inbounds i8** %output2.04.us, i32 %64
  br label %.lr.ph6.split.us

.lr.ph.us:
  %var3.02.us = phi i32 [ %75, %.lr.ph.us ], [ %44, %.lr.ph6.split.us ], [ %44, %59 ]
  %xindex.01.us = phi i32 [ %76, %.lr.ph.us ], [ 0, %.lr.ph6.split.us ], [ 0, %59 ]
  %70 = add nsw i32 %xindex.01.us, %var4.15.us
  %71 = getelementptr inbounds [10 x [64 x i16]*]* %23, i32 0, i32 %70
  %72 = load [64 x i16]** %71, align 4
  %73 = getelementptr inbounds [64 x i16]* %72, i32 0, i32 0
  tail call void %37(%struct.ref2* %cinfo, %struct.ref3* %32, i16* %73, i8** %output2.04.us, i32 %var3.02.us) nounwind
  %74 = load i32* %40, align 4
  %75 = add i32 %74, %var3.02.us
  %76 = add nsw i32 %xindex.01.us, 1
  %exitcond = icmp eq i32 %76, %39
  br i1 %exitcond, label %.loopexit.us, label %.lr.ph.us

.loopexit.us:
  %.pre23.pre = load i32* %45, align 4
  br label %._crit_edge27

._crit_edge26:
  %var4.15 = phi i32 [ %var4.09, %.lr.ph6..lr.ph6.split_crit_edge ], [ %78, %._crit_edge26 ]
  %var5.03 = phi i32 [ 0, %.lr.ph6..lr.ph6.split_crit_edge ], [ %79, %._crit_edge26 ]
  %77 = load i32* %52, align 4
  %78 = add nsw i32 %77, %var4.15
  %79 = add nsw i32 %var5.03, 1
  %80 = icmp slt i32 %79, %46
  br i1 %80, label %._crit_edge26, label %._crit_edge

._crit_edge:
  %split = phi i32 [ %66, %._crit_edge27 ], [ %78, %._crit_edge26 ]
  %.pre25 = load i32* %19, align 4
  br label %81

; <label>:81
  %82 = phi i32 [ %.pre25, %._crit_edge ], [ %30, %.lr.ph10 ]
  %var4.1.lcssa = phi i32 [ %split, %._crit_edge ], [ %var4.09, %.lr.ph10 ]
  %83 = add nsw i32 %ci.08, 1
  %84 = icmp slt i32 %83, %82
  br i1 %84, label %.lr.ph10, label %._crit_edge11

._crit_edge11:
  %85 = phi i32 [ %28, %.preheader7 ], [ %82, %81 ]
  %86 = add i32 %var2.012, 1
  %87 = icmp ugt i32 %86, %5
  br i1 %87, label %._crit_edge14, label %.preheader7

._crit_edge14:
  %.pre21 = load i32* %13, align 4
  br label %88

; <label>:88
  %89 = phi i32 [ %.pre21, %._crit_edge14 ], [ %25, %24 ]
  store void (%struct.ref2*)* null, void (%struct.ref2*)** %18, align 4
  %90 = add nsw i32 %var1.015, 1
  %91 = icmp slt i32 %90, %89
  br i1 %91, label %24, label %._crit_edge19

._crit_edge19:
  ret i32 3
}

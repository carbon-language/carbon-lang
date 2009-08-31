; RUN: llvm-as < %s | opt -inline -argpromotion -instcombine -disable-output

; This test was failing because the inliner would inline @list_DeleteElement
; into @list_DeleteDuplicates and then into @inf_GetBackwardPartnerLits,
; turning the indirect call into a direct one.  This allowed instcombine to see
; the bitcast and eliminate it, deleting the original call and introducing
; another one.  This crashed the inliner because the new call was not in the
; callgraph.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"


define void @list_DeleteElement(i32 (i8*, i8*)* nocapture %Test) nounwind ssp {
entry:
  %0 = call i32 %Test(i8* null, i8* undef) nounwind
  ret void
}


define void @list_DeleteDuplicates(i32 (i8*, i8*)* nocapture %Test) nounwind ssp {
foo:
  call void @list_DeleteElement(i32 (i8*, i8*)* %Test) nounwind ssp 
  call fastcc void @list_Rplacd1284() nounwind ssp
  unreachable

}

define internal i32 @inf_LiteralsHaveSameSubtermAndAreFromSameClause(i32* nocapture %L1, i32* nocapture %L2) nounwind readonly ssp {
entry:
  unreachable
}


define internal fastcc void @inf_GetBackwardPartnerLits(i32* nocapture %Flags) nounwind ssp {
test:
  call void @list_DeleteDuplicates(i32 (i8*, i8*)* bitcast (i32 (i32*, i32*)* @inf_LiteralsHaveSameSubtermAndAreFromSameClause to i32 (i8*, i8*)*)) nounwind 
  ret void
}


define void @inf_BackwardEmptySortPlusPlus() nounwind ssp {
entry:
  call fastcc void @inf_GetBackwardPartnerLits(i32* null) nounwind ssp
  unreachable
}

define void @inf_BackwardWeakening() nounwind ssp {
entry:
  call fastcc void @inf_GetBackwardPartnerLits(i32* null) nounwind ssp
  unreachable
}




declare fastcc void @list_Rplacd1284() nounwind ssp

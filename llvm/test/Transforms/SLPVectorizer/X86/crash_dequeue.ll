; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"
%"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731" = type { double*, double*, double*, double** }

; Function Attrs: nounwind ssp uwtable
define void @_ZSt6uniqueISt15_Deque_iteratorIdRdPdEET_S4_S4_(%"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* %__first, %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* nocapture %__last) {
entry:
  %_M_cur2.i.i = getelementptr inbounds %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731", %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* %__first, i64 0, i32 0
  %0 = load double** %_M_cur2.i.i, align 8
  %_M_first3.i.i = getelementptr inbounds %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731", %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* %__first, i64 0, i32 1
  %_M_cur2.i.i81 = getelementptr inbounds %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731", %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* %__last, i64 0, i32 0
  %1 = load double** %_M_cur2.i.i81, align 8
  %_M_first3.i.i83 = getelementptr inbounds %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731", %"struct.std::_Deque_iterator.4.157.174.208.259.276.344.731"* %__last, i64 0, i32 1
  %2 = load double** %_M_first3.i.i83, align 8
  br i1 undef, label %_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit, label %while.cond.i.preheader

while.cond.i.preheader:                           ; preds = %entry
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.body.i, %while.cond.i.preheader
  br i1 undef, label %_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit, label %while.body.i

while.body.i:                                     ; preds = %while.cond.i
  br i1 undef, label %_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit, label %while.cond.i

_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit: ; preds = %while.body.i, %while.cond.i, %entry
  %3 = phi double* [ %2, %entry ], [ %2, %while.cond.i ], [ undef, %while.body.i ]
  %4 = phi double* [ %0, %entry ], [ %1, %while.cond.i ], [ undef, %while.body.i ]
  store double* %4, double** %_M_cur2.i.i, align 8
  store double* %3, double** %_M_first3.i.i, align 8
  br i1 undef, label %if.then.i55, label %while.cond

if.then.i55:                                      ; preds = %_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %if.then.i55, %_ZSt13adjacent_findISt15_Deque_iteratorIdRdPdEET_S4_S4_.exit
  br label %while.cond
}

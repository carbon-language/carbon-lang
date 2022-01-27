; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-unknown"

; The subtract instruction %3 will be optimized (combined and predicated) with the select
; inside the loop.  In this case, the kill flag on the subtract should be removed or else
; it will fail verification.

%struct.PROOFSEARCH_HELP = type { %struct.LIST_HELP*, %struct.LIST_HELP*, %struct.LIST_HELP*, %struct.LIST_HELP*, %struct.SHARED_INDEX_NODE*, %struct.LIST_HELP*, %struct.SHARED_INDEX_NODE*, %struct.LIST_HELP*, %struct.SORTTHEORY_HELP*, %struct.SORTTHEORY_HELP*, %struct.SORTTHEORY_HELP*, %struct.SHARED_INDEX_NODE*, %struct.LIST_HELP*, i32*, i32*, %struct.LIST_HELP*, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.SORTTHEORY_HELP = type { %struct.st*, [4000 x %struct.NODE_HELP*], %struct.LIST_HELP*, %struct.LIST_HELP*, i32 }
%struct.st = type { %struct.subst*, %struct.LIST_HELP*, %struct.LIST_HELP*, i16, i16 }
%struct.subst = type { %struct.subst*, i32, %struct.term* }
%struct.term = type { i32, %union.anon, %struct.LIST_HELP*, i32, i32 }
%union.anon = type { %struct.LIST_HELP* }
%struct.NODE_HELP = type { %struct.LIST_HELP*, i32, i32, i32, %struct.LIST_HELP*, i32 }
%struct.SHARED_INDEX_NODE = type { %struct.st*, [3001 x %struct.term*], [4000 x %struct.term*], i32 }
%struct.LIST_HELP = type { %struct.LIST_HELP*, i8* }
%struct.CLAUSE_HELP = type { i32, i32, i32, i32, i32*, i32, %struct.LIST_HELP*, %struct.LIST_HELP*, i32, i32, %struct.LITERAL_HELP**, i32, i32, i32, i32 }
%struct.LITERAL_HELP = type { i32, i32, i32, %struct.CLAUSE_HELP*, %struct.term* }

declare void @foo(%struct.PROOFSEARCH_HELP*, %struct.CLAUSE_HELP*)

; CHECK-LABEL: @test
; CHECK: it
; CHECK-NEXT: sub

define hidden fastcc %struct.LIST_HELP* @test(%struct.PROOFSEARCH_HELP* %Search, %struct.LIST_HELP* %ClauseList, i32 %Level, %struct.LIST_HELP** nocapture %New) {
entry:
  %cmp4.i.i = icmp ugt i32 %Level, 31
  %0 = add i32 %Level, -32
  %1 = lshr i32 %0, 5
  %2 = shl nuw i32 %1, 5
  %3 = sub i32 %0, %2
  %4 = add nuw nsw i32 %1, 1
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %Scan.038 = phi %struct.LIST_HELP* [ %ClauseList, %entry ], [ %9, %for.inc ]
  %car.i33 = getelementptr inbounds %struct.LIST_HELP, %struct.LIST_HELP* %Scan.038, i32 0, i32 1
  %5 = bitcast i8** %car.i33 to %struct.CLAUSE_HELP**
  %6 = load %struct.CLAUSE_HELP*, %struct.CLAUSE_HELP** %5, align 4
  %. = add i32 %4, 10
  %.Level = select i1 %cmp4.i.i, i32 %3, i32 %Level
  %splitfield.i = getelementptr inbounds %struct.CLAUSE_HELP, %struct.CLAUSE_HELP* %6, i32 0, i32 4
  %7 = load i32*, i32** %splitfield.i, align 4
  %arrayidx.i = getelementptr inbounds i32, i32* %7, i32 %.
  %8 = load i32, i32* %arrayidx.i, align 4
  %shl.i = shl i32 1, %.Level
  %and.i = and i32 %8, %shl.i
  %cmp4.i = icmp eq i32 %and.i, 0
  br i1 %cmp4.i, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  tail call void @foo(%struct.PROOFSEARCH_HELP* %Search, %struct.CLAUSE_HELP* %6)
  store i8* null, i8** %car.i33, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %cdr.i = getelementptr inbounds %struct.LIST_HELP, %struct.LIST_HELP* %Scan.038, i32 0, i32 0
  %9 = load %struct.LIST_HELP*, %struct.LIST_HELP** %cdr.i, align 4
  br label %for.body
}

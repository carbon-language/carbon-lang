; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/dot-dumper.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t3 \
; RUN:  -r=%t1.bc,main,px \
; RUN:  -r=%t1.bc,main_alias,p \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,A, \
; RUN:  -r=%t2.bc,foo,p \
; RUN:  -r=%t2.bc,bar,p \
; RUN:  -r=%t2.bc,A,p \
; RUN:  -r=%t2.bc,B,p
; RUN: cat %t3.index.dot | FileCheck %s

; CHECK:        digraph Summary
; CHECK-NEXT:      Module:

; CHECK-LABEL:     subgraph cluster_0
; Node definitions can appear in any order, but they should go before edge list.
; CHECK-DAG:         M0_[[MAIN_ALIAS:[0-9]+]] [{{.*}}main_alias{{.*}}]; // alias, dead
; CHECK-DAG:         M0_[[MAIN:[0-9]+]] [{{.*}}main|extern{{.*}}]; // function
; CHECK:             // Edges:
; CHECK-NEXT:        M0_[[MAIN_ALIAS]] -> M0_[[MAIN]] [{{.*}}]; // alias

; CHECK-LABEL:     subgraph cluster_1 {
; CHECK:             M1_[[A:[0-9]+]] [{{.*}}A|extern{{.*}}]; // variable

; CHECK-DAG:         M1_[[FOO:[0-9]+]] [{{.*}}foo|extern{{.*}}]; // function, not eligible to import
; CHECK-DAG:         M1_[[B:[0-9]+]] [{{.*}}B|extern{{.*}}]; // variable
; CHECK-DAG:         M1_[[BAR:[0-9]+]] [{{.*}}bar|extern{{.*}}]; // function, dead
; CHECK:             Edges:

; Order of edges in dot file is undefined
; CHECK-DAG:         M1_[[FOO]] -> M1_[[B]] [{{.*}}]; // ref
; CHECK-DAG:         M1_[[FOO]] -> M1_[[A]] [{{.*}}]; // ref
; CHECK:           }

; Cross-module edges
; CHECK-DAG:       M0_[[MAIN]] -> M1_[[FOO]] // call
; CHECK-DAG:       M0_[[MAIN]] -> M1_[[A]] [{{.*}}]; // ref

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr {
  %1 = tail call i32 (...) @foo()
  %2 = load i32, i32* @A, align 4
  %3 = add nsw i32 %2, %1
  ret i32 %3
}
@main_alias = weak_odr alias i32 (), i32 ()* @main
declare i32 @foo(...) local_unnamed_addr

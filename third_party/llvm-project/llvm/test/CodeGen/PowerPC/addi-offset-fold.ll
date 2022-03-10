; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%struct.S0 = type <{ i32, [5 x i8] }>

; Function Attrs: norecurse nounwind readnone
define signext i32 @foo([2 x i64] %a.coerce) local_unnamed_addr #0 {
entry:
  %a = alloca %struct.S0, align 8
  %a.coerce.fca.0.extract = extractvalue [2 x i64] %a.coerce, 0
  %a.coerce.fca.1.extract = extractvalue [2 x i64] %a.coerce, 1
  %a.0.a.0..sroa_cast = bitcast %struct.S0* %a to i64*
  store i64 %a.coerce.fca.0.extract, i64* %a.0.a.0..sroa_cast, align 8
  %tmp.sroa.2.0.extract.trunc = trunc i64 %a.coerce.fca.1.extract to i8
  %a.8.a.8..sroa_idx = getelementptr inbounds %struct.S0, %struct.S0* %a, i64 0, i32 1, i64 4
  store i8 %tmp.sroa.2.0.extract.trunc, i8* %a.8.a.8..sroa_idx, align 8
  %a.4.a.4..sroa_idx = getelementptr inbounds %struct.S0, %struct.S0* %a, i64 0, i32 1
  %a.4.a.4..sroa_cast = bitcast [5 x i8]* %a.4.a.4..sroa_idx to i40*
  %a.4.a.4.bf.load = load i40, i40* %a.4.a.4..sroa_cast, align 4
  %bf.lshr = lshr i40 %a.4.a.4.bf.load, 31
  %bf.lshr.tr = trunc i40 %bf.lshr to i32
  %bf.cast = and i32 %bf.lshr.tr, 127
  ret i32 %bf.cast

; CHECK-LABEL: @foo
; FIXME: We don't need to do these stores at all.
; CHECK-DAG: std 3, -24(1)
; CHECK-DAG: stb 4, -16(1)
; CHECK-DAG: lwz [[REG2:[0-9]+]], -20(1)
; CHECK-DAG: rlwinm 3, [[REG2]], 1, 31, 31
; CHECK: rlwimi 3, 4, 1, 25, 30
; CHECK: blr
}

attributes #0 = { nounwind "frame-pointer"="all" "target-cpu"="ppc64le" }


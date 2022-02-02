; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; The SplitIndexingFromLoad tranformation exposed an isel backend bug.  This
; testcase used to generate stwx 4, 3, 64.  stwx does not have an
; immediate-offset format (note the 64) and it should not be matched.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%class.test = type { [64 x i8], [5 x i8] }

; CHECK-LABEL: f:
; CHECK-NOT: stwx {{[0-9]+}}, {{[0-9]+}}, 64
define void @f(%class.test* %this) {
entry:
  %Subminor.i.i = getelementptr inbounds %class.test, %class.test* %this, i64 0, i32 1
  %0 = bitcast [5 x i8]* %Subminor.i.i to i40*
  %bf.load2.i.i = load i40, i40* %0, align 4
  %bf.clear7.i.i = and i40 %bf.load2.i.i, -8589934592
  store i40 %bf.clear7.i.i, i40* %0, align 4
  ret void
}

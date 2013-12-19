; RUN: llc < %s -march=cpp | FileCheck %s

define void @f1(i8* byval, i8* inalloca) {
; CHECK: ByVal
; CHECK: InAlloca
  ret void
}

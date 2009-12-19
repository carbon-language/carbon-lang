; RUN: llvm-as < %s | llc -march=pic16 | FileCheck %s 

;CHECK: #include p16f1xxx.inc
;CHECK: #include stdmacros.inc

define void @foo() nounwind {
entry:
  ret void
}

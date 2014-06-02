; RUN: llc -mcpu=generic -march=x86 < %s | FileCheck %s
; PR19858

declare {i8, i1} @llvm.umul.with.overflow.i8(i8 %a, i8 %b)
define i8 @testumulo(i32 %argc) {
; CHECK: imulw
; CHECK: testb %{{.+}}, %{{.+}}
; CHECK: je [[NOOVERFLOWLABEL:.+]]
; CHECK: {{.*}}[[NOOVERFLOWLABEL]]:
; CHECK-NEXT: movb
; CHECK-NEXT: retl
top:
  %RHS = trunc i32 %argc to i8
  %umul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 25, i8 %RHS)
  %ex = extractvalue { i8, i1 } %umul, 1
  br i1 %ex, label %overflow, label %nooverlow

overflow:
  ret i8 %RHS

nooverlow:
  %umul.value = extractvalue { i8, i1 } %umul, 0
  ret i8 %umul.value
}

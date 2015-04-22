; RUN: llvm-link %s -override %S/Inputs/override-with-internal-linkage.ll -S | FileCheck %s
; RUN: llvm-link -override %S/Inputs/override-with-internal-linkage.ll %s -S | FileCheck %s

; CHECK-LABEL: define internal i32 @foo2
; CHECK-NEXT: entry:
; CHECK-NEXT: %add = add nsw i32 %i, %i
; CHECK-NEXT: ret i32 %add

; CHECK-LABEL: define i32 @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 4
define internal i32 @foo(i32 %i) {
entry:
  %add = add nsw i32 %i, %i
  ret i32 %add
}

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** %argv) {
entry:
  %a = call i32 @foo(i32 2)
  ret i32 %a
}

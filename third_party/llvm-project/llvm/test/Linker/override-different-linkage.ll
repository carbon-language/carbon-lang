; RUN: llvm-link %s -override %S/Inputs/override-different-linkage.ll -S | FileCheck %s
; RUN: llvm-link -override %S/Inputs/override-different-linkage.ll %s -S | FileCheck %s


; CHECK-LABEL: define linkonce i32 @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 4
define weak i32 @foo(i32 %i) {
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

; RUN: llc < %s -march=ppc32 -o %t
; RUN: grep sync %t
; RUN: not grep msync %t 
; RUN: llc < %s -march=ppc32 -mcpu=440 | grep msync

define i32 @has_a_fence(i32 %a, i32 %b) nounwind {
entry:
  fence acquire
  %cond = icmp eq i32 %a, %b
  br i1 %cond, label %IfEqual, label %IfUnequal

IfEqual:
  fence release
  br label %end

IfUnequal:
  fence release
  ret i32 0

end:
  ret i32 1
}


; RUN: echo {%bug_type = type opaque \
; RUN:     declare i32 @bug_a(%bug_type*) \
; RUN:     declare i32 @bug_b(%bug_type*) } > %t.ll
; RUN: llvm-link %t.ll %s
; PR11464

%bug_type = type { %bug_type* }
%bar = type { i32 }

define i32 @bug_a(%bug_type* %fp) nounwind uwtable {
entry:
  %d_stream = getelementptr inbounds %bug_type* %fp, i64 0, i32 0
  ret i32 0
}

define i32 @bug_b(%bar* %a) nounwind uwtable {
entry:
  ret i32 0
}

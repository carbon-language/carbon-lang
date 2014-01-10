; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/syntax.prof 2>&1 | FileCheck -check-prefix=NO-DEBUG %s
; RUN: not opt < %s -sample-profile -sample-profile-file=missing.prof 2>&1 | FileCheck -check-prefix=MISSING-FILE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_fn_header.prof 2>&1 | FileCheck -check-prefix=BAD-FN-HEADER %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_sample_line.prof 2>&1 | FileCheck -check-prefix=BAD-SAMPLE-LINE %s

define void @empty() {
entry:
  ret void
}
; NO-DEBUG: LLVM ERROR: No debug information found in function empty
; MISSING-FILE: LLVM ERROR: Could not open file missing.prof: No such file or directory
; BAD-FN-HEADER: LLVM ERROR: {{.*}}bad_fn_header.prof:1: Expected 'mangled_name:NUM:NUM', found empty:100:BAD
; BAD-SAMPLE-LINE: LLVM ERROR: {{.*}}bad_sample_line.prof:3: Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found 1: BAD

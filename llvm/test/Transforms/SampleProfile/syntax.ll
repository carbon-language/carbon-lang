; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/syntax.prof 2>&1 | FileCheck -check-prefix=NO-DEBUG %s
; RUN: not opt < %s -sample-profile -sample-profile-file=missing.prof 2>&1 | FileCheck -check-prefix=MISSING-FILE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/missing_symtab.prof 2>&1 | FileCheck -check-prefix=MISSING-SYMTAB %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/missing_num_syms.prof 2>&1 | FileCheck -check-prefix=MISSING-NUM-SYMS %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_fn_header.prof 2>&1 | FileCheck -check-prefix=BAD-FN-HEADER %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/bad_sample_line.prof 2>&1 | FileCheck -check-prefix=BAD-SAMPLE-LINE %s
; RUN: not opt < %s -sample-profile -sample-profile-file=%S/Inputs/missing_samples.prof 2>&1 | FileCheck -check-prefix=MISSING-SAMPLES %s

define void @empty() {
entry:
  ret void
}
; NO-DEBUG: LLVM ERROR: No debug information found in function empty
; MISSING-FILE: LLVM ERROR: Could not open file missing.prof: No such file or directory
; MISSING-SYMTAB: LLVM ERROR: {{.*}}missing_symtab.prof:1: Expected 'symbol table', found 1
; MISSING-NUM-SYMS: LLVM ERROR: {{.*}}missing_num_syms.prof:2: Expected a number, found empty
; BAD-FN-HEADER: LLVM ERROR: {{.*}}bad_fn_header.prof:4: Expected 'mangled_name:NUM:NUM:NUM', found empty:100:BAD
; BAD-SAMPLE-LINE: LLVM ERROR: {{.*}}bad_sample_line.prof:6: Expected 'mangled_name:NUM:NUM:NUM', found 1: BAD
; MISSING-SAMPLES: LLVM ERROR: {{.*}}missing_samples.prof:6: Unexpected end of file

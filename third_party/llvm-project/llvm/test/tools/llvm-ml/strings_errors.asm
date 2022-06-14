; RUN: not llvm-ml -m64 -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.code

oversize_string_test PROC

  mov rax, "abcdefghi"
  mov rax, 'abcdefghi'
; CHECK: error: literal value out of range
; CHECK: error: literal value out of range

  ret
oversize_string_test ENDP

end

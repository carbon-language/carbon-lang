; This is not an assembly file, this is just to run the test.
; The test verifies that llvm-ar produces a binary output.

; FIXME: They malform LF into CRLF. Investigating.
; XFAIL: mingw32,win32

;RUN: llvm-ar p %p/Inputs/GNU.a very_long_bytecode_file_name.bc | cmp -s %p/Inputs/very_long_bytecode_file_name.bc -

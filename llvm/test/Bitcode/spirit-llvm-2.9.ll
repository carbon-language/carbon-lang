; This isn't really an assembly file. It just runs the test on the bitcode to 
; ensure bitcode file backward compatibility.  No need for FileCheck as the 
; BitcodeReader will fail with an assert if broken. This test case was 
; generated using a clang binary, based on LLVM 2.9, downloaded from llvm.org.
; RUN: llvm-dis < %s.bc > /dev/null

; REQUIRES: shell

; This first line will generate the .o files for the next run line
; RUN: %lli_mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -enable-cache-manager %s

; This line tests MCJIT object loading
; RUN: %lli_mcjit -extra-object=%p/Inputs/multi-module-b.o -extra-object=%p/Inputs/multi-module-c.o %s

; These lines put the object files into an archive
; RUN: llvm-ar r %p/Inputs/load-object.a %p/Inputs/multi-module-b.o
; RUN: llvm-ar r %p/Inputs/load-object.a %p/Inputs/multi-module-c.o

; This line test MCJIT archive loading
; RUN: %lli_mcjit -extra-archive=%p/Inputs/load-object.a %s

; These lines clean up our temporary files
; RUN: rm -f %p/Inputs/load-object-a.o
; RUN: rm -f %p/Inputs/multi-module-b.o
; RUN: rm -f %p/Inputs/multi-module-c.o
; RUN: rm -f %p/Inputs/load-object.a

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

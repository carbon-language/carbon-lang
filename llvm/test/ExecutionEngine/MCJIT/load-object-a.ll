; REQUIRES: shell

; This first line will generate the .o files for the next run line
; RUN: mkdir -p %t.cachedir
; RUN: %lli_mcjit -extra-module=%p/Inputs/multi-module-b.ll -extra-module=%p/Inputs/multi-module-c.ll -enable-cache-manager -object-cache-dir=%t.cachedir %s

; This line tests MCJIT object loading
; RUN: %lli_mcjit -extra-object=%t.cachedir/%p/Inputs/multi-module-b.o -extra-object=%t.cachedir/%p/Inputs/multi-module-c.o %s

; These lines put the object files into an archive
; RUN: llvm-ar r %t.cachedir/%p/Inputs/load-object.a %t.cachedir/%p/Inputs/multi-module-b.o
; RUN: llvm-ar r %t.cachedir/%p/Inputs/load-object.a %t.cachedir/%p/Inputs/multi-module-c.o

; This line test MCJIT archive loading
; RUN: %lli_mcjit -extra-archive=%t.cachedir/%p/Inputs/load-object.a %s

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}

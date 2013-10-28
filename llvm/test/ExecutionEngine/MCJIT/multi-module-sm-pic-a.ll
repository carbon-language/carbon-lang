; RUN: %lli_mcjit -extra-module=%p/multi-module-b.ir -extra-module=%p/multi-module-c.ir -relocation-model=pic -code-model=small %s > /dev/null
; XFAIL: mips, i686, i386, aarch64, arm

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}


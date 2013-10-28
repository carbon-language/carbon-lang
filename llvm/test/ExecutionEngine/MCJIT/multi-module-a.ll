; RUN: %lli_mcjit -extra-module=%p/multi-module-b.ir -extra-module=%p/multi-module-c.ir %s > /dev/null

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}


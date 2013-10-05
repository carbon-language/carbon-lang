; RUN: %lli_mcjit -extra-modules=%p/multi-module-b.ir,%p/multi-module-c.ir -relocation-model=pic -code-model=small %s > /dev/null

declare i32 @FB()

define i32 @main() {
  %r = call i32 @FB( )   ; <i32> [#uses=1]
  ret i32 %r
}


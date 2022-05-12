; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -bugpoint-crashfuncattr -silence-passes
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck -check-prefixes=ALL,ENABLED %s
; RUN: bugpoint -disable-attribute-remove -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -bugpoint-crashfuncattr -silence-passes
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck -check-prefixes=ALL,DISABLED %s

; REQUIRES: plugins

; ALL: f() #[[ATTRS:[0-9]+]]
define void @f() #0 {
  ret void
}

; ENABLED: attributes #[[ATTRS]] = { "bugpoint-crash" }
; DISABLED: attributes #[[ATTRS]] = { noinline "bugpoint-crash" "frame-pointer"="non-leaf" }
attributes #0 = { noinline "bugpoint-crash" "frame-pointer"="non-leaf" }
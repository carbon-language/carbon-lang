; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

define dso_local void @func() {
entry:
  ret void
}

define dso_local void @main() {
entry:
  call void asm sideeffect inteldialect "call ${0:P}", "*m,~{dirflag},~{fpsr},~{flags}"(void ()* elementtype(void ()) @func)
  ret void
; CHECK-LABEL: main:
; CHECK: {{## InlineAsm Start|#APP}}
; CHECK: {{call(l|q) func$}}
; CHECK: {{## InlineAsm End|#NO_APP}}
; CHECK: ret{{l|q}}
}

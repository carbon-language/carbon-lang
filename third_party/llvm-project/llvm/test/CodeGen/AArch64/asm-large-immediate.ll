; RUN: llc < %s -mtriple=aarch64-eabi -no-integrated-as | FileCheck %s

define void @test() {
entry:
; CHECK: /* result: 68719476738 */
        tail call void asm sideeffect "/* result: ${0:c} */", "i,~{dirflag},~{fpsr},~{flags}"( i64 68719476738 )
; CHECK: /* result: -68719476738 */
        tail call void asm sideeffect "/* result: ${0:n} */", "i,~{dirflag},~{fpsr},~{flags}"( i64 68719476738 )
        ret void
}

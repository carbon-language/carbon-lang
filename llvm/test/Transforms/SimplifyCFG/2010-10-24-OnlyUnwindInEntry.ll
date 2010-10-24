; RUN: opt %s -simplifycfg -disable-output
; PR8445

define void @test() {
      unwind
}

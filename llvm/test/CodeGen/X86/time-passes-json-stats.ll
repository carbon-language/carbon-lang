; RUN: llc -mtriple=x86_64-- -stats-json=true -stats -time-passes %s -o /dev/null 2>&1 | FileCheck %s

; Verify that we use the argument pass name instead of the full name as a json
; key for timers.
;
; CHECK: {
; CHECK-NEXT: "asm-printer.EmittedInsts":
; CHECK-NOT: Virtual Register Map
; CHECK: "time.pass.virtregmap.wall":
; CHECK: "time.pass.virtregmap.user":
; CHECK: "time.pass.virtregmap.sys":
; CHECK: Virtual Register Map

define void @test_stats() { ret void }

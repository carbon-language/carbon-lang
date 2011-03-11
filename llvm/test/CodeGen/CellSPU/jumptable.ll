;RUN: llc --march=cellspu %s -o - | FileCheck %s
; This is to check that emitting jumptables doesn't crash llc
define i32 @test(i32 %param) {
entry:
;CHECK:        ai      {{\$.}}, $3, -1
;CHECK:        clgti   {{\$., \$.}}, 3
;CHECK:        brnz    {{\$.}},.LBB0_2
  switch i32 %param, label %bb1 [
    i32 1, label %bb3
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb1
  ]

bb1:                                            
  ret i32 1
bb2:      
  ret i32 2
bb3:     
  ret i32 3
}

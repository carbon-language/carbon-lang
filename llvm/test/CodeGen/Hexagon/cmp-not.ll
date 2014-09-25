; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate matching compare insn.

; Function Attrs: nounwind
define i32 @neqi(i32 %argc) #0 {
entry:
  %p = alloca i8, align 1
  %0 = tail call i1 @llvm.hexagon.C4.cmpneqi(i32 %argc, i32 512)
  %conv = zext i1 %0 to i8
  store volatile i8 %conv, i8* %p, align 1
  %p.0.p.0. = load volatile i8* %p, align 1
  %conv1 = zext i8 %p.0.p.0. to i32
  ret i32 %conv1
}
; CHECK:	p{{[0-3]}}{{ *}} = !cmp.eq(r{{[0-9]+}}, ##512)

; Function Attrs: nounwind readnone
declare i1 @llvm.hexagon.C4.cmpneqi(i32, i32) #1

; Function Attrs: nounwind
define i32 @ngti(i32 %argc) #0 {
entry:
  %p = alloca i8, align 1
  %0 = tail call i1 @llvm.hexagon.C4.cmpltei(i32 %argc, i32 4)
  %conv = zext i1 %0 to i8
  store volatile i8 %conv, i8* %p, align 1
  %p.0.p.0. = load volatile i8* %p, align 1
  %conv1 = zext i8 %p.0.p.0. to i32
  ret i32 %conv1
}
; CHECK:	p{{[0-3]}}{{ *}} = !cmp.gt(r{{[0-9]+}}, #4)

; Function Attrs: nounwind readnone
declare i1 @llvm.hexagon.C4.cmpltei(i32, i32) #1

; Function Attrs: nounwind
define i32 @ngtui(i32 %argc) #0 {
entry:
  %p = alloca i8, align 1
  %0 = tail call i1 @llvm.hexagon.C4.cmplteui(i32 %argc, i32 4)
  %conv = zext i1 %0 to i8
  store volatile i8 %conv, i8* %p, align 1
  %p.0.p.0. = load volatile i8* %p, align 1
  %conv1 = zext i8 %p.0.p.0. to i32
  ret i32 %conv1
}
; CHECK: 	p{{[0-3]}}{{ *}} = !cmp.gtu(r{{[0-9]+}}, #4)

; Function Attrs: nounwind readnone
declare i1 @llvm.hexagon.C4.cmplteui(i32, i32) #1

; RUN: opt -mtriple i386 -Os -S %s -o - | FileCheck %s
; RUN: opt -mtriple x86_64 -Os -S %s -o - | FileCheck %s
; RUN: opt -mtriple armv7 -Os -S %s -o - | FileCheck %s

%struct.D16 = type { [16 x double] }

declare void @_Z2OpP3D16PKS_S2_(%struct.D16*, %struct.D16*, %struct.D16*)

define void @_Z7TestRefRK3D16S1_(%struct.D16* noalias sret %agg.result, %struct.D16* %RHS, %struct.D16* %LHS) {
  %1 = alloca %struct.D16*, align 8
  %2 = alloca %struct.D16*, align 8
  store %struct.D16* %RHS, %struct.D16** %1, align 8
  store %struct.D16* %LHS, %struct.D16** %2, align 8
  %3 = load %struct.D16** %1, align 8
  %4 = load %struct.D16** %2, align 8
  call void @_Z2OpP3D16PKS_S2_(%struct.D16* %agg.result, %struct.D16* %3, %struct.D16* %4)
  ret void
}

; CHECK: define void @_Z7TestRefRK3D16S1_({{.*}}) {
; CHECK:   tail call void @_Z2OpP3D16PKS_S2_(%struct.D16* %agg.result, %struct.D16* %RHS, %struct.D16* %LHS)
; CHECK:   ret void
; CHECK: }

define void @_Z7TestVal3D16S_(%struct.D16* noalias sret %agg.result, %struct.D16* byval align 8 %RHS, %struct.D16* byval align 8 %LHS) {
  call void @_Z2OpP3D16PKS_S2_(%struct.D16* %agg.result, %struct.D16* %RHS, %struct.D16* %LHS)
  ret void
}

; CHECK: define void @_Z7TestVal3D16S_({{.*}}) {
; CHECK:   tail call void @_Z2OpP3D16PKS_S2_(%struct.D16* %agg.result, %struct.D16* %RHS, %struct.D16* %LHS)
; CHECK:   ret void
; CHECK: }


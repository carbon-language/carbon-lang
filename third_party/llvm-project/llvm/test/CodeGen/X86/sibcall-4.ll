; RUN: llc < %s -mtriple=i386-pc-linux-gnu | FileCheck %s
; pr7610

define ghccc void @t(i32* %Base_Arg, i32* %Sp_Arg, i32* %Hp_Arg, i32 %R1_Arg) nounwind {
cm1:
; CHECK-LABEL: t:
; CHECK: jmpl *%eax
  %nm3 = getelementptr i32, i32* %Sp_Arg, i32 1
  %nm9 = load i32, i32* %Sp_Arg
  %nma = inttoptr i32 %nm9 to void (i32*, i32*, i32*, i32)*
  tail call ghccc void %nma(i32* %Base_Arg, i32* %nm3, i32* %Hp_Arg, i32 %R1_Arg) nounwind
  ret void
}

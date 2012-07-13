; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; NB: currently tests only vclz, should also test vcnt and vcls

define <4 x i16> @vclz16(<4 x i16>* %A) nounwind {
;CHECK: @vclz16
        %tmp1 = load <4 x i16>* %A
        %tmp2 = call <4 x i16> @llvm.arm.neon.vclz.v4i16(<4 x i16> %tmp1)
;CHECK: {{call.*@llvm.ctlz.v4i16\(<4 x i16>.*, i1 false}}
        ret <4 x i16> %tmp2
}

declare <4 x i16>  @llvm.arm.neon.vclz.v4i16(<4 x i16>) nounwind readnone

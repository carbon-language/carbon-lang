; RUN: llvm-upgrade < %s | llvm-as | opt -indvars -disable-output

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8"
	%struct.vorbis_dsp_state = type { int, %struct.vorbis_info*, float**, float**, int, int, int, int, int, int, int, int, int, long, long, long, long, long, long, sbyte* }
	%struct.vorbis_info = type { int, int, int, int, int, int, int, sbyte* }

implementation   ; Functions:

void %_ve_envelope_search() {
entry:
	br bool false, label %cond_true27, label %bb137

cond_true27:		; preds = %entry
	br bool false, label %cond_true52, label %bb80

cond_true52:		; preds = %cond_true27
	%tmp152.i = bitcast float 0.000000e+00 to uint		; <uint> [#uses=1]
	br label %cond_next182.i

cond_next182.i:		; preds = %cond_next182.i, %cond_true52
	%decay.i.0 = phi uint [ %tmp195.i, %cond_next182.i ], [ %tmp152.i, %cond_true52 ]		; <uint> [#uses=1]
	%tmp194.i53 = bitcast uint %decay.i.0 to float		; <float> [#uses=1]
	%tmp195.i = sub float %tmp194.i53, 8.000000e+00		; <float> [#uses=1]
	%tmp195.i = bitcast float %tmp195.i to uint		; <uint> [#uses=1]
	br bool false, label %cond_next182.i, label %bb418.i.preheader

bb418.i.preheader:		; preds = %cond_next182.i
	ret void

bb80:		; preds = %cond_true27
	ret void

bb137:		; preds = %entry
	ret void
}

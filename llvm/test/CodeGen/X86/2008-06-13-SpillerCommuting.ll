; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -relocation-model=pic -stats |& grep {spiller       - Number of instructions commuted}

	%struct.CABAC_context_element = type { i8, i8 }
	%struct.MB_Info_CABAC = type { i8, i8, [2 x i8], i8, i8, i8, i16, i16, [4 x i8], [8 x %struct.MotionVector] }
	%struct.MotionVector = type { i16, i16 }
	%struct.RBSP2 = type { i32, i32, i16, i8, i16, i16, <1 x i64>, i32, i32, i32*, i32*, i32*, i32*, i32, i32, i32, i32, i32, i8, i16, i8, %struct.MB_Info_CABAC*, %struct.MB_Info_CABAC*, [2 x %struct.MB_Info_CABAC], [12 x i8], [460 x %struct.CABAC_context_element], [10 x i8], [10 x i8], [10 x i16], [4 x [120 x i32]], [15 x [36 x i8]], [6 x [8 x i8]], i16* }
	%struct.Slice_Info = type { i32, i8, %struct.seq_parameter_set_rbsp_t*, %struct.seq_parameter_set_rbsp_t, i32, i16*, i8, i8, i8, i8, i16, i32 }
	%struct.seq_parameter_set_rbsp_t = type { i32, i32, i32 }
@_ZL21CABAC_CTX_state_table = external constant [64 x i16]		; <[64 x i16]*> [#uses=1]
@_ZL15rLPS_table_64x4 = external constant [64 x [4 x i8]]		; <[64 x [4 x i8]]*> [#uses=1]

define i32 @_ZN5RBSP220residual_block_cabacEP10Slice_InfoP13MB_Info_CABACS3_hjhhbPtPs(%struct.RBSP2* %this, %struct.Slice_Info* %slice, %struct.MB_Info_CABAC* %up, %struct.MB_Info_CABAC* %left, i8 zeroext  %maxNumCoeff, i32 %blk_i, i8 zeroext  %iCbCr, i8 zeroext  %ctxBlockCat, i8 zeroext  %intra_flag, i16* %mask, i16* %res) nounwind  {
entry:
	%tmp43.i1590 = getelementptr %struct.RBSP2* %this, i32 0, i32 0		; <i32*> [#uses=1]
	br label %bb803

bb803:		; preds = %_ZN5RBSP211decode_1bitEP21CABAC_context_element.exit1581, %entry
	%numCoeff.11749 = phi i32 [ 0, %entry ], [ %numCoeff.11749.tmp868, %_ZN5RBSP211decode_1bitEP21CABAC_context_element.exit1581 ]		; <i32> [#uses=1]
	%tmp28.i1503 = load i8* null, align 1		; <i8> [#uses=1]
	%tmp30.i1504 = getelementptr %struct.RBSP2* %this, i32 0, i32 25, i32 0, i32 0		; <i8*> [#uses=2]
	%tmp31.i1505 = load i8* %tmp30.i1504, align 1		; <i8> [#uses=1]
	%tmp3233.i1506 = zext i8 %tmp31.i1505 to i32		; <i32> [#uses=2]
	%tmp35.i1507 = getelementptr [64 x i16]* @_ZL21CABAC_CTX_state_table, i32 0, i32 %tmp3233.i1506		; <i16*> [#uses=1]
	%tmp36.i1508 = load i16* %tmp35.i1507, align 2		; <i16> [#uses=1]
	%tmp363738.i1509 = zext i16 %tmp36.i1508 to i32		; <i32> [#uses=1]
	%tmp51.i1514 = getelementptr [64 x [4 x i8]]* @_ZL15rLPS_table_64x4, i32 0, i32 %tmp3233.i1506, i32 0		; <i8*> [#uses=1]
	%tmp52.i1515 = load i8* %tmp51.i1514, align 1		; <i8> [#uses=1]
	%tmp5758.i1516 = zext i8 %tmp52.i1515 to i32		; <i32> [#uses=1]
	%tmp60.i1517 = sub i32 0, %tmp5758.i1516		; <i32> [#uses=1]
	store i32 %tmp60.i1517, i32* %tmp43.i1590, align 16
	br i1 false, label %_ZN5RBSP211decode_1bitEP21CABAC_context_element.exit1581, label %bb.i1537

bb.i1537:		; preds = %bb803
	unreachable

_ZN5RBSP211decode_1bitEP21CABAC_context_element.exit1581:		; preds = %bb803
	%tmp328329.i1580 = trunc i32 %tmp363738.i1509 to i8		; <i8> [#uses=1]
	store i8 %tmp328329.i1580, i8* %tmp30.i1504, align 1
	%toBool865 = icmp eq i8 %tmp28.i1503, 0		; <i1> [#uses=1]
	%numCoeff.11749.tmp868 = select i1 %toBool865, i32 %numCoeff.11749, i32 0		; <i32> [#uses=1]
	br label %bb803
}

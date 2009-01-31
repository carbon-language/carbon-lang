; RUN: llvm-as < %s | opt -instcombine | llvm-dis
; PR3452
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@__clz_tab = external constant [256 x i8]		; <[256 x i8]*> [#uses=3]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (void (i128, i128, i128*)* @__udivmodti4 to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc void @__udivmodti4(i128 %n, i128 %d, i128* %rp) nounwind alwaysinline {
entry:
	%tmp1 = trunc i128 %n to i64		; <i64> [#uses=12]
	%sroa.store.elt = lshr i128 %n, 64		; <i128> [#uses=1]
	%tmp2 = trunc i128 %sroa.store.elt to i64		; <i64> [#uses=12]
	%tmp3 = trunc i128 %d to i64		; <i64> [#uses=12]
	%sroa.store.elt16 = lshr i128 %d, 64		; <i128> [#uses=1]
	%tmp4 = trunc i128 %sroa.store.elt16 to i64		; <i64> [#uses=7]
	%tmp5 = icmp eq i64 %tmp4, 0		; <i1> [#uses=1]
	br i1 %tmp5, label %bb, label %bb86

bb:		; preds = %entry
	%tmp6 = icmp ugt i64 %tmp3, %tmp2		; <i1> [#uses=1]
	br i1 %tmp6, label %bb33.thread, label %bb50

bb33.thread:		; preds = %bb
	br label %bb31

bb31:		; preds = %bb33, %bb33.thread
	%__a28.0.reg2mem.0 = phi i64 [ 56, %bb33.thread ], [ %tmp9, %bb33 ]		; <i64> [#uses=3]
	%.cast = and i64 %__a28.0.reg2mem.0, 4294967288		; <i64> [#uses=1]
	%tmp = shl i64 255, %.cast		; <i64> [#uses=1]
	%tmp7 = and i64 %tmp3, %tmp		; <i64> [#uses=1]
	%tmp8 = icmp eq i64 %tmp7, 0		; <i1> [#uses=1]
	br i1 %tmp8, label %bb33, label %bb34

bb33:		; preds = %bb31
	%tmp9 = add i64 %__a28.0.reg2mem.0, -8		; <i64> [#uses=3]
	%tmp10 = icmp eq i64 %tmp9, 0		; <i1> [#uses=1]
	br i1 %tmp10, label %bb34, label %bb31

bb34:		; preds = %bb33, %bb31
	%__a28.0.reg2mem.1 = phi i64 [ %__a28.0.reg2mem.0, %bb31 ], [ %tmp9, %bb33 ]		; <i64> [#uses=2]
	%.cast35 = and i64 %__a28.0.reg2mem.1, 4294967288		; <i64> [#uses=1]
	%tmp11 = lshr i64 %tmp3, %.cast35		; <i64> [#uses=1]
	%tmp12 = getelementptr [256 x i8]* @__clz_tab, i64 0, i64 %tmp11		; <i8*> [#uses=1]
	%tmp13 = load i8* %tmp12, align 1		; <i8> [#uses=1]
	%tmp14 = zext i8 %tmp13 to i64		; <i64> [#uses=1]
	%tmp15 = add i64 %tmp14, %__a28.0.reg2mem.1		; <i64> [#uses=2]
	%tmp16 = sub i64 64, %tmp15		; <i64> [#uses=7]
	%tmp17 = icmp eq i64 %tmp15, 64		; <i1> [#uses=1]
	br i1 %tmp17, label %bb41, label %bb36

bb36:		; preds = %bb34
	%.cast37 = and i64 %tmp16, 4294967295		; <i64> [#uses=1]
	%tmp18 = shl i64 %tmp3, %.cast37		; <i64> [#uses=1]
	%.cast38 = and i64 %tmp16, 4294967295		; <i64> [#uses=1]
	%tmp19 = shl i64 %tmp2, %.cast38		; <i64> [#uses=1]
	%tmp20 = sub i64 64, %tmp16		; <i64> [#uses=1]
	%.cast39 = and i64 %tmp20, 4294967295		; <i64> [#uses=1]
	%tmp21 = lshr i64 %tmp1, %.cast39		; <i64> [#uses=1]
	%tmp22 = or i64 %tmp19, %tmp21		; <i64> [#uses=1]
	%.cast40 = and i64 %tmp16, 4294967295		; <i64> [#uses=1]
	%tmp23 = shl i64 %tmp1, %.cast40		; <i64> [#uses=1]
	br label %bb41

bb41:		; preds = %bb36, %bb34
	%n1.0 = phi i64 [ %tmp2, %bb34 ], [ %tmp22, %bb36 ]		; <i64> [#uses=2]
	%n0.0 = phi i64 [ %tmp1, %bb34 ], [ %tmp23, %bb36 ]		; <i64> [#uses=2]
	%d0.0 = phi i64 [ %tmp3, %bb34 ], [ %tmp18, %bb36 ]		; <i64> [#uses=8]
	%tmp24 = lshr i64 %d0.0, 32		; <i64> [#uses=4]
	%tmp25 = and i64 %d0.0, 4294967295		; <i64> [#uses=2]
	%tmp26 = urem i64 %n1.0, %tmp24		; <i64> [#uses=1]
	%tmp27 = udiv i64 %n1.0, %tmp24		; <i64> [#uses=1]
	%tmp28 = mul i64 %tmp27, %tmp25		; <i64> [#uses=3]
	%tmp29 = shl i64 %tmp26, 32		; <i64> [#uses=1]
	%tmp30 = lshr i64 %n0.0, 32		; <i64> [#uses=1]
	%tmp31 = or i64 %tmp29, %tmp30		; <i64> [#uses=3]
	%tmp32 = icmp ult i64 %tmp31, %tmp28		; <i1> [#uses=1]
	br i1 %tmp32, label %bb42, label %bb45

bb42:		; preds = %bb41
	%tmp33 = add i64 %tmp31, %d0.0		; <i64> [#uses=4]
	%.not = icmp uge i64 %tmp33, %d0.0		; <i1> [#uses=1]
	%tmp34 = icmp ult i64 %tmp33, %tmp28		; <i1> [#uses=1]
	%or.cond = and i1 %tmp34, %.not		; <i1> [#uses=1]
	br i1 %or.cond, label %bb44, label %bb45

bb44:		; preds = %bb42
	%tmp35 = add i64 %tmp33, %d0.0		; <i64> [#uses=1]
	br label %bb45

bb45:		; preds = %bb44, %bb42, %bb41
	%__r123.0 = phi i64 [ %tmp31, %bb41 ], [ %tmp33, %bb42 ], [ %tmp35, %bb44 ]		; <i64> [#uses=1]
	%tmp36 = sub i64 %__r123.0, %tmp28		; <i64> [#uses=2]
	%tmp37 = urem i64 %tmp36, %tmp24		; <i64> [#uses=1]
	%tmp38 = udiv i64 %tmp36, %tmp24		; <i64> [#uses=1]
	%tmp39 = mul i64 %tmp38, %tmp25		; <i64> [#uses=5]
	%tmp40 = shl i64 %tmp37, 32		; <i64> [#uses=1]
	%tmp41 = and i64 %n0.0, 4294967295		; <i64> [#uses=1]
	%tmp42 = or i64 %tmp40, %tmp41		; <i64> [#uses=3]
	%tmp43 = icmp ult i64 %tmp42, %tmp39		; <i1> [#uses=1]
	br i1 %tmp43, label %bb46, label %bb83

bb46:		; preds = %bb45
	%tmp44 = add i64 %tmp42, %d0.0		; <i64> [#uses=4]
	%.not137 = icmp uge i64 %tmp44, %d0.0		; <i1> [#uses=1]
	%tmp45 = icmp ult i64 %tmp44, %tmp39		; <i1> [#uses=1]
	%or.cond138 = and i1 %tmp45, %.not137		; <i1> [#uses=1]
	br i1 %or.cond138, label %bb48, label %bb83

bb48:		; preds = %bb46
	%tmp46 = add i64 %tmp44, %d0.0		; <i64> [#uses=1]
	br label %bb83

bb50:		; preds = %bb
	%tmp47 = icmp eq i64 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp47, label %bb51, label %bb56.thread

bb51:		; preds = %bb50
	%tmp48 = udiv i64 1, %tmp3		; <i64> [#uses=1]
	br label %bb56.thread

bb53:		; preds = %bb56, %bb56.thread
	%__a19.0.reg2mem.0 = phi i64 [ 56, %bb56.thread ], [ %tmp51, %bb56 ]		; <i64> [#uses=3]
	%.cast54 = and i64 %__a19.0.reg2mem.0, 4294967288		; <i64> [#uses=1]
	%tmp133 = shl i64 255, %.cast54		; <i64> [#uses=1]
	%tmp49 = and i64 %d0.1.ph, %tmp133		; <i64> [#uses=1]
	%tmp50 = icmp eq i64 %tmp49, 0		; <i1> [#uses=1]
	br i1 %tmp50, label %bb56, label %bb57

bb56.thread:		; preds = %bb51, %bb50
	%d0.1.ph = phi i64 [ %tmp48, %bb51 ], [ %tmp3, %bb50 ]		; <i64> [#uses=5]
	br label %bb53

bb56:		; preds = %bb53
	%tmp51 = add i64 %__a19.0.reg2mem.0, -8		; <i64> [#uses=3]
	%tmp52 = icmp eq i64 %tmp51, 0		; <i1> [#uses=1]
	br i1 %tmp52, label %bb57, label %bb53

bb57:		; preds = %bb56, %bb53
	%__a19.0.reg2mem.1 = phi i64 [ %__a19.0.reg2mem.0, %bb53 ], [ %tmp51, %bb56 ]		; <i64> [#uses=2]
	%.cast58 = and i64 %__a19.0.reg2mem.1, 4294967288		; <i64> [#uses=1]
	%tmp53 = lshr i64 %d0.1.ph, %.cast58		; <i64> [#uses=1]
	%tmp54 = getelementptr [256 x i8]* @__clz_tab, i64 0, i64 %tmp53		; <i8*> [#uses=1]
	%tmp55 = load i8* %tmp54, align 1		; <i8> [#uses=1]
	%tmp56 = zext i8 %tmp55 to i64		; <i64> [#uses=1]
	%tmp57 = add i64 %tmp56, %__a19.0.reg2mem.1		; <i64> [#uses=2]
	%tmp58 = sub i64 64, %tmp57		; <i64> [#uses=7]
	%tmp59 = icmp eq i64 %tmp57, 64		; <i1> [#uses=1]
	br i1 %tmp59, label %bb74, label %bb60

bb60:		; preds = %bb57
	%tmp60 = sub i64 64, %tmp58		; <i64> [#uses=2]
	%.cast61 = and i64 %tmp58, 4294967295		; <i64> [#uses=1]
	%tmp61 = shl i64 %d0.1.ph, %.cast61		; <i64> [#uses=9]
	%.cast62 = and i64 %tmp60, 4294967295		; <i64> [#uses=1]
	%tmp62 = lshr i64 %tmp2, %.cast62		; <i64> [#uses=2]
	%.cast63 = and i64 %tmp58, 4294967295		; <i64> [#uses=1]
	%tmp63 = shl i64 %tmp2, %.cast63		; <i64> [#uses=1]
	%.cast64 = and i64 %tmp60, 4294967295		; <i64> [#uses=1]
	%tmp64 = lshr i64 %tmp1, %.cast64		; <i64> [#uses=1]
	%tmp65 = or i64 %tmp63, %tmp64		; <i64> [#uses=2]
	%.cast65 = and i64 %tmp58, 4294967295		; <i64> [#uses=1]
	%tmp66 = shl i64 %tmp1, %.cast65		; <i64> [#uses=1]
	%tmp67 = lshr i64 %tmp61, 32		; <i64> [#uses=4]
	%tmp68 = and i64 %tmp61, 4294967295		; <i64> [#uses=2]
	%tmp69 = urem i64 %tmp62, %tmp67		; <i64> [#uses=1]
	%tmp70 = udiv i64 %tmp62, %tmp67		; <i64> [#uses=1]
	%tmp71 = mul i64 %tmp70, %tmp68		; <i64> [#uses=3]
	%tmp72 = shl i64 %tmp69, 32		; <i64> [#uses=1]
	%tmp73 = lshr i64 %tmp65, 32		; <i64> [#uses=1]
	%tmp74 = or i64 %tmp72, %tmp73		; <i64> [#uses=3]
	%tmp75 = icmp ult i64 %tmp74, %tmp71		; <i1> [#uses=1]
	br i1 %tmp75, label %bb66, label %bb69

bb66:		; preds = %bb60
	%tmp76 = add i64 %tmp74, %tmp61		; <i64> [#uses=4]
	%.not139 = icmp uge i64 %tmp76, %tmp61		; <i1> [#uses=1]
	%tmp77 = icmp ult i64 %tmp76, %tmp71		; <i1> [#uses=1]
	%or.cond140 = and i1 %tmp77, %.not139		; <i1> [#uses=1]
	br i1 %or.cond140, label %bb68, label %bb69

bb68:		; preds = %bb66
	%tmp78 = add i64 %tmp76, %tmp61		; <i64> [#uses=1]
	br label %bb69

bb69:		; preds = %bb68, %bb66, %bb60
	%__r114.0 = phi i64 [ %tmp74, %bb60 ], [ %tmp76, %bb66 ], [ %tmp78, %bb68 ]		; <i64> [#uses=1]
	%tmp79 = sub i64 %__r114.0, %tmp71		; <i64> [#uses=2]
	%tmp80 = urem i64 %tmp79, %tmp67		; <i64> [#uses=1]
	%tmp81 = udiv i64 %tmp79, %tmp67		; <i64> [#uses=1]
	%tmp82 = mul i64 %tmp81, %tmp68		; <i64> [#uses=3]
	%tmp83 = shl i64 %tmp80, 32		; <i64> [#uses=1]
	%tmp84 = and i64 %tmp65, 4294967295		; <i64> [#uses=1]
	%tmp85 = or i64 %tmp83, %tmp84		; <i64> [#uses=3]
	%tmp86 = icmp ult i64 %tmp85, %tmp82		; <i1> [#uses=1]
	br i1 %tmp86, label %bb70, label %bb73

bb70:		; preds = %bb69
	%tmp87 = add i64 %tmp85, %tmp61		; <i64> [#uses=4]
	%.not141 = icmp uge i64 %tmp87, %tmp61		; <i1> [#uses=1]
	%tmp88 = icmp ult i64 %tmp87, %tmp82		; <i1> [#uses=1]
	%or.cond142 = and i1 %tmp88, %.not141		; <i1> [#uses=1]
	br i1 %or.cond142, label %bb72, label %bb73

bb72:		; preds = %bb70
	%tmp89 = add i64 %tmp87, %tmp61		; <i64> [#uses=1]
	br label %bb73

bb73:		; preds = %bb72, %bb70, %bb69
	%__r013.0 = phi i64 [ %tmp85, %bb69 ], [ %tmp87, %bb70 ], [ %tmp89, %bb72 ]		; <i64> [#uses=1]
	br label %bb74

bb74:		; preds = %bb73, %bb57
	%__r013.0.pn = phi i64 [ %__r013.0, %bb73 ], [ %tmp2, %bb57 ]		; <i64> [#uses=1]
	%.pn135 = phi i64 [ %tmp82, %bb73 ], [ %d0.1.ph, %bb57 ]		; <i64> [#uses=1]
	%n0.2 = phi i64 [ %tmp66, %bb73 ], [ %tmp1, %bb57 ]		; <i64> [#uses=2]
	%d0.2 = phi i64 [ %tmp61, %bb73 ], [ %d0.1.ph, %bb57 ]		; <i64> [#uses=8]
	%n1.1 = sub i64 %__r013.0.pn, %.pn135		; <i64> [#uses=2]
	%tmp90 = lshr i64 %d0.2, 32		; <i64> [#uses=4]
	%tmp91 = and i64 %d0.2, 4294967295		; <i64> [#uses=2]
	%tmp92 = urem i64 %n1.1, %tmp90		; <i64> [#uses=1]
	%tmp93 = udiv i64 %n1.1, %tmp90		; <i64> [#uses=1]
	%tmp94 = mul i64 %tmp93, %tmp91		; <i64> [#uses=3]
	%tmp95 = shl i64 %tmp92, 32		; <i64> [#uses=1]
	%tmp96 = lshr i64 %n0.2, 32		; <i64> [#uses=1]
	%tmp97 = or i64 %tmp95, %tmp96		; <i64> [#uses=3]
	%tmp98 = icmp ult i64 %tmp97, %tmp94		; <i1> [#uses=1]
	br i1 %tmp98, label %bb75, label %bb78

bb75:		; preds = %bb74
	%tmp99 = add i64 %tmp97, %d0.2		; <i64> [#uses=4]
	%.not143 = icmp uge i64 %tmp99, %d0.2		; <i1> [#uses=1]
	%tmp100 = icmp ult i64 %tmp99, %tmp94		; <i1> [#uses=1]
	%or.cond144 = and i1 %tmp100, %.not143		; <i1> [#uses=1]
	br i1 %or.cond144, label %bb77, label %bb78

bb77:		; preds = %bb75
	%tmp101 = add i64 %tmp99, %d0.2		; <i64> [#uses=1]
	br label %bb78

bb78:		; preds = %bb77, %bb75, %bb74
	%__r17.0 = phi i64 [ %tmp97, %bb74 ], [ %tmp99, %bb75 ], [ %tmp101, %bb77 ]		; <i64> [#uses=1]
	%tmp102 = sub i64 %__r17.0, %tmp94		; <i64> [#uses=2]
	%tmp103 = urem i64 %tmp102, %tmp90		; <i64> [#uses=1]
	%tmp104 = udiv i64 %tmp102, %tmp90		; <i64> [#uses=1]
	%tmp105 = mul i64 %tmp104, %tmp91		; <i64> [#uses=5]
	%tmp106 = shl i64 %tmp103, 32		; <i64> [#uses=1]
	%tmp107 = and i64 %n0.2, 4294967295		; <i64> [#uses=1]
	%tmp108 = or i64 %tmp106, %tmp107		; <i64> [#uses=3]
	%tmp109 = icmp ult i64 %tmp108, %tmp105		; <i1> [#uses=1]
	br i1 %tmp109, label %bb79, label %bb83

bb79:		; preds = %bb78
	%tmp110 = add i64 %tmp108, %d0.2		; <i64> [#uses=4]
	%.not145 = icmp uge i64 %tmp110, %d0.2		; <i1> [#uses=1]
	%tmp111 = icmp ult i64 %tmp110, %tmp105		; <i1> [#uses=1]
	%or.cond146 = and i1 %tmp111, %.not145		; <i1> [#uses=1]
	br i1 %or.cond146, label %bb81, label %bb83

bb81:		; preds = %bb79
	%tmp112 = add i64 %tmp110, %d0.2		; <i64> [#uses=1]
	br label %bb83

bb83:		; preds = %bb81, %bb79, %bb78, %bb48, %bb46, %bb45
	%bm.0 = phi i64 [ %tmp16, %bb46 ], [ %tmp16, %bb48 ], [ %tmp16, %bb45 ], [ %tmp58, %bb79 ], [ %tmp58, %bb81 ], [ %tmp58, %bb78 ]		; <i64> [#uses=1]
	%__r06.0.pn = phi i64 [ %tmp42, %bb45 ], [ %tmp44, %bb46 ], [ %tmp46, %bb48 ], [ %tmp108, %bb78 ], [ %tmp110, %bb79 ], [ %tmp112, %bb81 ]		; <i64> [#uses=1]
	%.pn = phi i64 [ %tmp39, %bb46 ], [ %tmp39, %bb48 ], [ %tmp39, %bb45 ], [ %tmp105, %bb79 ], [ %tmp105, %bb81 ], [ %tmp105, %bb78 ]		; <i64> [#uses=1]
	%tmp113 = icmp eq i128* %rp, null		; <i1> [#uses=1]
	br i1 %tmp113, label %bb131, label %bb84

bb84:		; preds = %bb83
	%n0.1 = sub i64 %__r06.0.pn, %.pn		; <i64> [#uses=1]
	%.cast85 = and i64 %bm.0, 4294967295		; <i64> [#uses=1]
	%tmp114 = lshr i64 %n0.1, %.cast85		; <i64> [#uses=1]
	%tmp115 = zext i64 %tmp114 to i128		; <i128> [#uses=1]
	store i128 %tmp115, i128* %rp, align 16
	br label %bb131

bb86:		; preds = %entry
	%tmp116 = icmp ugt i64 %tmp4, %tmp2		; <i1> [#uses=1]
	br i1 %tmp116, label %bb87, label %bb93.thread

bb93.thread:		; preds = %bb86
	br label %bb90

bb87:		; preds = %bb86
	%tmp117 = icmp eq i128* %rp, null		; <i1> [#uses=1]
	br i1 %tmp117, label %bb131, label %bb88

bb88:		; preds = %bb87
	store i128 %n, i128* %rp, align 16
	br label %bb131

bb90:		; preds = %bb93, %bb93.thread
	%__a.0.reg2mem.0 = phi i64 [ 56, %bb93.thread ], [ %tmp120, %bb93 ]		; <i64> [#uses=3]
	%.cast91 = and i64 %__a.0.reg2mem.0, 4294967288		; <i64> [#uses=1]
	%tmp136 = shl i64 255, %.cast91		; <i64> [#uses=1]
	%tmp118 = and i64 %tmp4, %tmp136		; <i64> [#uses=1]
	%tmp119 = icmp eq i64 %tmp118, 0		; <i1> [#uses=1]
	br i1 %tmp119, label %bb93, label %bb94

bb93:		; preds = %bb90
	%tmp120 = add i64 %__a.0.reg2mem.0, -8		; <i64> [#uses=3]
	%tmp121 = icmp eq i64 %tmp120, 0		; <i1> [#uses=1]
	br i1 %tmp121, label %bb94, label %bb90

bb94:		; preds = %bb93, %bb90
	%__a.0.reg2mem.1 = phi i64 [ %__a.0.reg2mem.0, %bb90 ], [ %tmp120, %bb93 ]		; <i64> [#uses=2]
	%.cast95 = and i64 %__a.0.reg2mem.1, 4294967288		; <i64> [#uses=1]
	%tmp122 = lshr i64 %tmp4, %.cast95		; <i64> [#uses=1]
	%tmp123 = getelementptr [256 x i8]* @__clz_tab, i64 0, i64 %tmp122		; <i8*> [#uses=1]
	%tmp124 = load i8* %tmp123, align 1		; <i8> [#uses=1]
	%tmp125 = zext i8 %tmp124 to i64		; <i64> [#uses=1]
	%tmp126 = add i64 %tmp125, %__a.0.reg2mem.1		; <i64> [#uses=2]
	%tmp127 = sub i64 64, %tmp126		; <i64> [#uses=7]
	%tmp128 = icmp eq i64 %tmp126, 64		; <i1> [#uses=1]
	br i1 %tmp128, label %bb96, label %bb103

bb96:		; preds = %bb94
	%tmp129 = icmp ugt i64 %tmp2, %tmp4		; <i1> [#uses=1]
	%tmp130 = icmp uge i64 %tmp1, %tmp3		; <i1> [#uses=1]
	%tmp131 = or i1 %tmp129, %tmp130		; <i1> [#uses=1]
	br i1 %tmp131, label %bb99, label %bb101

bb99:		; preds = %bb96
	%tmp132 = sub i64 %tmp1, %tmp3		; <i64> [#uses=2]
	%tmp134 = sub i64 %tmp2, %tmp4		; <i64> [#uses=1]
	%tmp135 = icmp ugt i64 %tmp132, %tmp1		; <i1> [#uses=1]
	%tmp137 = zext i1 %tmp135 to i64		; <i64> [#uses=1]
	%tmp138 = sub i64 %tmp134, %tmp137		; <i64> [#uses=1]
	br label %bb101

bb101:		; preds = %bb99, %bb96
	%tmp139 = phi i64 [ %tmp138, %bb99 ], [ %tmp2, %bb96 ]		; <i64> [#uses=1]
	%n0.3 = phi i64 [ %tmp132, %bb99 ], [ %tmp1, %bb96 ]		; <i64> [#uses=1]
	%tmp140 = icmp eq i128* %rp, null		; <i1> [#uses=1]
	br i1 %tmp140, label %bb131, label %bb102

bb102:		; preds = %bb101
	%tmp141 = zext i64 %n0.3 to i128		; <i128> [#uses=1]
	%tmp142 = zext i64 %tmp139 to i128		; <i128> [#uses=1]
	%tmp143 = shl i128 %tmp142, 64		; <i128> [#uses=1]
	%tmp144 = or i128 %tmp143, %tmp141		; <i128> [#uses=1]
	store i128 %tmp144, i128* %rp, align 16
	br label %bb131

bb103:		; preds = %bb94
	%tmp145 = sub i64 64, %tmp127		; <i64> [#uses=4]
	%.cast104 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp146 = shl i64 %tmp4, %.cast104		; <i64> [#uses=1]
	%.cast105 = and i64 %tmp145, 4294967295		; <i64> [#uses=1]
	%tmp147 = lshr i64 %tmp3, %.cast105		; <i64> [#uses=1]
	%tmp148 = or i64 %tmp146, %tmp147		; <i64> [#uses=9]
	%.cast106 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp149 = shl i64 %tmp3, %.cast106		; <i64> [#uses=3]
	%.cast107 = and i64 %tmp145, 4294967295		; <i64> [#uses=1]
	%tmp150 = lshr i64 %tmp2, %.cast107		; <i64> [#uses=2]
	%.cast108 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp151 = shl i64 %tmp2, %.cast108		; <i64> [#uses=1]
	%.cast109 = and i64 %tmp145, 4294967295		; <i64> [#uses=1]
	%tmp152 = lshr i64 %tmp1, %.cast109		; <i64> [#uses=1]
	%tmp153 = or i64 %tmp151, %tmp152		; <i64> [#uses=2]
	%.cast110 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp154 = shl i64 %tmp1, %.cast110		; <i64> [#uses=3]
	%tmp155 = lshr i64 %tmp148, 32		; <i64> [#uses=4]
	%tmp156 = and i64 %tmp148, 4294967295		; <i64> [#uses=2]
	%tmp157 = urem i64 %tmp150, %tmp155		; <i64> [#uses=1]
	%tmp158 = udiv i64 %tmp150, %tmp155		; <i64> [#uses=4]
	%tmp159 = mul i64 %tmp158, %tmp156		; <i64> [#uses=3]
	%tmp160 = shl i64 %tmp157, 32		; <i64> [#uses=1]
	%tmp161 = lshr i64 %tmp153, 32		; <i64> [#uses=1]
	%tmp162 = or i64 %tmp160, %tmp161		; <i64> [#uses=3]
	%tmp163 = icmp ult i64 %tmp162, %tmp159		; <i1> [#uses=1]
	br i1 %tmp163, label %bb111, label %bb114

bb111:		; preds = %bb103
	%tmp164 = add i64 %tmp158, -1		; <i64> [#uses=1]
	%tmp165 = add i64 %tmp162, %tmp148		; <i64> [#uses=4]
	%.not147 = icmp uge i64 %tmp165, %tmp148		; <i1> [#uses=1]
	%tmp166 = icmp ult i64 %tmp165, %tmp159		; <i1> [#uses=1]
	%or.cond148 = and i1 %tmp166, %.not147		; <i1> [#uses=1]
	br i1 %or.cond148, label %bb113, label %bb114

bb113:		; preds = %bb111
	%tmp167 = add i64 %tmp158, -2		; <i64> [#uses=1]
	%tmp168 = add i64 %tmp165, %tmp148		; <i64> [#uses=1]
	br label %bb114

bb114:		; preds = %bb113, %bb111, %bb103
	%__q1.0 = phi i64 [ %tmp158, %bb103 ], [ %tmp164, %bb111 ], [ %tmp167, %bb113 ]		; <i64> [#uses=1]
	%__r1.0 = phi i64 [ %tmp162, %bb103 ], [ %tmp165, %bb111 ], [ %tmp168, %bb113 ]		; <i64> [#uses=1]
	%tmp169 = sub i64 %__r1.0, %tmp159		; <i64> [#uses=2]
	%tmp170 = urem i64 %tmp169, %tmp155		; <i64> [#uses=1]
	%tmp171 = udiv i64 %tmp169, %tmp155		; <i64> [#uses=4]
	%tmp172 = mul i64 %tmp171, %tmp156		; <i64> [#uses=3]
	%tmp173 = shl i64 %tmp170, 32		; <i64> [#uses=1]
	%tmp174 = and i64 %tmp153, 4294967295		; <i64> [#uses=1]
	%tmp175 = or i64 %tmp173, %tmp174		; <i64> [#uses=3]
	%tmp176 = icmp ult i64 %tmp175, %tmp172		; <i1> [#uses=1]
	br i1 %tmp176, label %bb115, label %bb118

bb115:		; preds = %bb114
	%tmp177 = add i64 %tmp171, -1		; <i64> [#uses=1]
	%tmp178 = add i64 %tmp175, %tmp148		; <i64> [#uses=4]
	%.not149 = icmp uge i64 %tmp178, %tmp148		; <i1> [#uses=1]
	%tmp179 = icmp ult i64 %tmp178, %tmp172		; <i1> [#uses=1]
	%or.cond150 = and i1 %tmp179, %.not149		; <i1> [#uses=1]
	br i1 %or.cond150, label %bb117, label %bb118

bb117:		; preds = %bb115
	%tmp180 = add i64 %tmp171, -2		; <i64> [#uses=1]
	%tmp181 = add i64 %tmp178, %tmp148		; <i64> [#uses=1]
	br label %bb118

bb118:		; preds = %bb117, %bb115, %bb114
	%__q0.0 = phi i64 [ %tmp171, %bb114 ], [ %tmp177, %bb115 ], [ %tmp180, %bb117 ]		; <i64> [#uses=2]
	%__r0.0 = phi i64 [ %tmp175, %bb114 ], [ %tmp178, %bb115 ], [ %tmp181, %bb117 ]		; <i64> [#uses=1]
	%tmp182 = sub i64 %__r0.0, %tmp172		; <i64> [#uses=3]
	%tmp183 = shl i64 %__q1.0, 32		; <i64> [#uses=1]
	%tmp184 = or i64 %tmp183, %__q0.0		; <i64> [#uses=1]
	%tmp185 = and i64 %__q0.0, 4294967295		; <i64> [#uses=2]
	%tmp186 = lshr i64 %tmp184, 32		; <i64> [#uses=2]
	%tmp187 = and i64 %tmp149, 4294967295		; <i64> [#uses=2]
	%tmp188 = lshr i64 %tmp149, 32		; <i64> [#uses=2]
	%tmp189 = mul i64 %tmp185, %tmp187		; <i64> [#uses=2]
	%tmp190 = mul i64 %tmp185, %tmp188		; <i64> [#uses=1]
	%tmp191 = mul i64 %tmp186, %tmp187		; <i64> [#uses=2]
	%tmp192 = mul i64 %tmp186, %tmp188		; <i64> [#uses=1]
	%tmp193 = lshr i64 %tmp189, 32		; <i64> [#uses=1]
	%tmp194 = add i64 %tmp193, %tmp190		; <i64> [#uses=1]
	%tmp195 = add i64 %tmp194, %tmp191		; <i64> [#uses=3]
	%tmp196 = icmp ult i64 %tmp195, %tmp191		; <i1> [#uses=1]
	%tmp197 = select i1 %tmp196, i64 4294967296, i64 0		; <i64> [#uses=1]
	%__x3.0 = add i64 %tmp192, %tmp197		; <i64> [#uses=1]
	%tmp198 = lshr i64 %tmp195, 32		; <i64> [#uses=1]
	%tmp199 = add i64 %tmp198, %__x3.0		; <i64> [#uses=4]
	%tmp200 = shl i64 %tmp195, 32		; <i64> [#uses=1]
	%tmp201 = and i64 %tmp189, 4294967295		; <i64> [#uses=1]
	%tmp202 = or i64 %tmp200, %tmp201		; <i64> [#uses=4]
	%tmp203 = icmp ugt i64 %tmp199, %tmp182		; <i1> [#uses=1]
	br i1 %tmp203, label %bb125, label %bb121

bb121:		; preds = %bb118
	%tmp204 = icmp eq i64 %tmp199, %tmp182		; <i1> [#uses=1]
	%tmp205 = icmp ugt i64 %tmp202, %tmp154		; <i1> [#uses=1]
	%tmp206 = and i1 %tmp204, %tmp205		; <i1> [#uses=1]
	br i1 %tmp206, label %bb125, label %bb126

bb125:		; preds = %bb121, %bb118
	%tmp207 = sub i64 %tmp202, %tmp149		; <i64> [#uses=2]
	%tmp208 = sub i64 %tmp199, %tmp148		; <i64> [#uses=1]
	%tmp209 = icmp ugt i64 %tmp207, %tmp202		; <i1> [#uses=1]
	%tmp210 = zext i1 %tmp209 to i64		; <i64> [#uses=1]
	%tmp211 = sub i64 %tmp208, %tmp210		; <i64> [#uses=1]
	br label %bb126

bb126:		; preds = %bb125, %bb121
	%m1.0 = phi i64 [ %tmp199, %bb121 ], [ %tmp211, %bb125 ]		; <i64> [#uses=1]
	%m0.0 = phi i64 [ %tmp202, %bb121 ], [ %tmp207, %bb125 ]		; <i64> [#uses=1]
	%tmp212 = icmp eq i128* %rp, null		; <i1> [#uses=1]
	br i1 %tmp212, label %bb131, label %bb127

bb127:		; preds = %bb126
	%tmp213 = sub i64 %tmp154, %m0.0		; <i64> [#uses=2]
	%tmp214 = sub i64 %tmp182, %m1.0		; <i64> [#uses=1]
	%tmp215 = icmp ugt i64 %tmp213, %tmp154		; <i1> [#uses=1]
	%tmp216 = zext i1 %tmp215 to i64		; <i64> [#uses=1]
	%tmp217 = sub i64 %tmp214, %tmp216		; <i64> [#uses=2]
	%.cast128 = and i64 %tmp145, 4294967295		; <i64> [#uses=1]
	%tmp218 = shl i64 %tmp217, %.cast128		; <i64> [#uses=1]
	%.cast129 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp219 = lshr i64 %tmp213, %.cast129		; <i64> [#uses=1]
	%tmp220 = or i64 %tmp218, %tmp219		; <i64> [#uses=1]
	%.cast130 = and i64 %tmp127, 4294967295		; <i64> [#uses=1]
	%tmp221 = lshr i64 %tmp217, %.cast130		; <i64> [#uses=1]
	%tmp222 = zext i64 %tmp220 to i128		; <i128> [#uses=1]
	%tmp223 = zext i64 %tmp221 to i128		; <i128> [#uses=1]
	%tmp224 = shl i128 %tmp223, 64		; <i128> [#uses=1]
	%tmp225 = or i128 %tmp224, %tmp222		; <i128> [#uses=1]
	store i128 %tmp225, i128* %rp, align 16
	br label %bb131

bb131:		; preds = %bb127, %bb126, %bb102, %bb101, %bb88, %bb87, %bb84, %bb83
	ret void
}

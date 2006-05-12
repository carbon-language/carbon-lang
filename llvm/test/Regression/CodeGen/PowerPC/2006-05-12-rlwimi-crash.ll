; RUN: llvm-as < %s | llc -march=ppc32

	%struct.attr_desc = type { sbyte*, %struct.attr_desc*, %struct.attr_value*, %struct.attr_value*, uint }
	%struct.attr_value = type { %struct.rtx_def*, %struct.attr_value*, %struct.insn_ent*, int, int }
	%struct.insn_def = type { %struct.insn_def*, %struct.rtx_def*, int, int, int, int, int }
	%struct.insn_ent = type { %struct.insn_ent*, %struct.insn_def* }
	%struct.rtx_def = type { ushort, ubyte, ubyte, %struct.u }
	%struct.u = type { [1 x long] }

implementation   ; Functions:

void %find_attr() {
entry:
	%tmp26 = seteq %struct.attr_desc* null, null		; <bool> [#uses=1]
	br bool %tmp26, label %bb30, label %cond_true27

cond_true27:		; preds = %entry
	ret void

bb30:		; preds = %entry
	%tmp67 = seteq %struct.attr_desc* null, null		; <bool> [#uses=1]
	br bool %tmp67, label %cond_next92, label %cond_true68

cond_true68:		; preds = %bb30
	ret void

cond_next92:		; preds = %bb30
	%tmp173 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=2]
	%tmp174 = load uint* %tmp173		; <uint> [#uses=1]
	%tmp177 = and uint %tmp174, 4294967287		; <uint> [#uses=1]
	store uint %tmp177, uint* %tmp173
	%tmp180 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=1]
	%tmp181 = load uint* %tmp180		; <uint> [#uses=1]
	%tmp185 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=2]
	%tmp186 = load uint* %tmp185		; <uint> [#uses=1]
	%tmp183187 = shl uint %tmp181, ubyte 1		; <uint> [#uses=1]
	%tmp188 = and uint %tmp183187, 16		; <uint> [#uses=1]
	%tmp190 = and uint %tmp186, 4294967279		; <uint> [#uses=1]
	%tmp191 = or uint %tmp190, %tmp188		; <uint> [#uses=1]
	store uint %tmp191, uint* %tmp185
	%tmp193 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=1]
	%tmp194 = load uint* %tmp193		; <uint> [#uses=1]
	%tmp198 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=2]
	%tmp199 = load uint* %tmp198		; <uint> [#uses=1]
	%tmp196200 = shl uint %tmp194, ubyte 2		; <uint> [#uses=1]
	%tmp201 = and uint %tmp196200, 64		; <uint> [#uses=1]
	%tmp203 = and uint %tmp199, 4294967231		; <uint> [#uses=1]
	%tmp204 = or uint %tmp203, %tmp201		; <uint> [#uses=1]
	store uint %tmp204, uint* %tmp198
	%tmp206 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=1]
	%tmp207 = load uint* %tmp206		; <uint> [#uses=1]
	%tmp211 = getelementptr %struct.attr_desc* null, int 0, uint 4		; <uint*> [#uses=2]
	%tmp212 = load uint* %tmp211		; <uint> [#uses=1]
	%tmp209213 = shl uint %tmp207, ubyte 1		; <uint> [#uses=1]
	%tmp214 = and uint %tmp209213, 128		; <uint> [#uses=1]
	%tmp216 = and uint %tmp212, 4294967167		; <uint> [#uses=1]
	%tmp217 = or uint %tmp216, %tmp214		; <uint> [#uses=1]
	store uint %tmp217, uint* %tmp211
	ret void
}

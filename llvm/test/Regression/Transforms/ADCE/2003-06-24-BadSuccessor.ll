; RUN: as < %s | opt -adce -disable-output

target endian = little
target pointersize = 32
	%struct..CppObjTypeDesc = type { uint, ushort, ushort }
	%struct..TypeToken = type { uint, ushort, ushort }

implementation   ; Functions:

uint %C_ReFaxToDb() {
entry:		; No predecessors!
	br bool false, label %endif.0, label %then.0

then.0:		; preds = %entry
	ret uint 0

endif.0:		; preds = %entry
	br bool false, label %then.11, label %then.4

then.4:		; preds = %endif.0
	ret uint 0

then.11:		; preds = %endif.0
	br bool false, label %loopentry.0, label %else.2

loopentry.0:		; preds = %then.11, %endif.14, %loopentry.1
	br bool false, label %endif.14, label %loopexit.0

endif.14:		; preds = %loopentry.0
	br bool false, label %loopentry.1, label %loopentry.0

loopentry.1:		; preds = %endif.14, %then.53, %then.53, %then.53, %then.53, %then.53
	%SubArrays.10 = phi uint* [ %SubArrays.8, %then.53 ] , [ null, %endif.14 ]		; <uint*> [#uses=3]
	br bool false, label %no_exit.1, label %loopentry.0

no_exit.1:		; preds = %loopentry.1
	switch uint 0, label %label.17 [
		 uint 2, label %label.11
		 uint 19, label %label.10
	]

label.10:		; preds = %no_exit.1
	br bool false, label %then.43, label %endif.43

then.43:		; preds = %label.10
	br bool false, label %then.44, label %endif.44

then.44:		; preds = %then.43
	br bool false, label %shortcirc_next.4, label %endif.45

shortcirc_next.4:		; preds = %then.44
	br bool false, label %no_exit.2, label %loopexit.2

no_exit.2:		; preds = %shortcirc_next.4
	%tmp.897 = getelementptr uint* %SubArrays.10, long 0		; <uint*> [#uses=1]
	%tmp.899 = load uint* %tmp.897		; <uint> [#uses=1]
	store uint %tmp.899, uint* null
	ret uint 0

loopexit.2:		; preds = %shortcirc_next.4
	ret uint 0

endif.45:		; preds = %then.44
	ret uint 0

endif.44:		; preds = %then.43
	ret uint 0

endif.43:		; preds = %label.10
	ret uint 0

label.11:		; preds = %no_exit.1
	ret uint 0

label.17:		; preds = %no_exit.1, %no_exit.1, %no_exit.1, %no_exit.1, %no_exit.1, %no_exit.1
	br bool false, label %then.53, label %shortcirc_next.7

shortcirc_next.7:		; preds = %label.17
	br bool false, label %then.53, label %shortcirc_next.8

shortcirc_next.8:		; preds = %shortcirc_next.7
	ret uint 0

then.53:		; preds = %shortcirc_next.7, %label.17
	%SubArrays.8 = phi uint* [ %SubArrays.10, %shortcirc_next.7 ], [ %SubArrays.10, %label.17 ]		; <uint*> [#uses=5]
	%tmp.1023 = load uint* null		; <uint> [#uses=1]
	switch uint %tmp.1023, label %loopentry.1 []

loopexit.0:		; preds = %loopentry.0
	ret uint 0

else.2:		; preds = %then.11
	ret uint 0
}

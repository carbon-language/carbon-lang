; RUN: llvm-as < %s | opt -dse

"java/lang/Object" = type { %struct.llvm_java_object_base }
	"java/lang/StringBuffer" = type { "java/lang/Object", int, { "java/lang/Object", uint, [0 x ushort] }*, bool }
	%struct.llvm_java_object_base = type opaque

implementation   ; Functions:

void "java/lang/StringBuffer/ensureCapacity_unsynchronized(I)V"() {
bc0:
	%tmp = getelementptr "java/lang/StringBuffer"* null, int 0, uint 3		; <bool*> [#uses=1]
	br bool false, label %bc16, label %bc7

bc16:		; preds = %bc0
	%tmp91 = getelementptr "java/lang/StringBuffer"* null, int 0, uint 2		; <{ "java/lang/Object", uint, [0 x ushort] }**> [#uses=1]
	store { "java/lang/Object", uint, [0 x ushort] }* null, { "java/lang/Object", uint, [0 x ushort] }** %tmp91
	store bool false, bool* %tmp
	ret void

bc7:		; preds = %bc0
	ret void
}

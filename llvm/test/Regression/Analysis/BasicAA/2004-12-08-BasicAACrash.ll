; RUN: llvm-as < %s | opt -licm

"java/lang/Object" = type { %struct.llvm_java_object_base }
	"java/lang/StringBuffer" = type { "java/lang/Object", int, { "java/lang/Object", uint, [0 x ushort] }*, bool }
	%struct.llvm_java_object_base = type opaque

implementation   ; Functions:

void "java/lang/StringBuffer/setLength(I)V"(%struct.llvm_java_object_base*) {
bc0:
	br bool false, label %bc40, label %bc65

bc65:		; preds = %bc0, %bc40
	ret void

bc40:		; preds = %bc0, %bc40
	%tmp75 = cast %struct.llvm_java_object_base* %0 to "java/lang/StringBuffer"*		; <"java/lang/StringBuffer"*> [#uses=1]
	%tmp76 = getelementptr "java/lang/StringBuffer"* %tmp75, int 0, uint 1		; <int*> [#uses=1]
	store int 0, int* %tmp76
	%tmp381 = cast %struct.llvm_java_object_base* %0 to "java/lang/StringBuffer"*		; <"java/lang/StringBuffer"*> [#uses=1]
	%tmp392 = getelementptr "java/lang/StringBuffer"* %tmp381, int 0, uint 1		; <int*> [#uses=1]
	%tmp403 = load int* %tmp392		; <int> [#uses=0]
	br bool false, label %bc40, label %bc65
}

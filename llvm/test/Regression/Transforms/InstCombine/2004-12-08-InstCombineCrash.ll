; RUN: llvm-as %s -o - | opt -instcombine

	%struct.llvm_java_object_base = type opaque
	"java/lang/Object" = type { %struct.llvm_java_object_base }
	"java/lang/StringBuffer" = type { "java/lang/Object", int, { "java/lang/Object", uint, [0 x ushort] }*, bool }

implementation   ; Functions:

void "java/lang/StringBuffer/append(Ljava/lang/String;)Ljava/lang/StringBuffer;"() {
bc0:
	%tmp53 = getelementptr "java/lang/StringBuffer"* null, int 0, uint 1		; <int*> [#uses=1]
	store int 0, int* %tmp53
	ret void
}

; RUN: llvm-as < %s | opt -basicaa -aa-eval -disable-output

	%struct..RefPoint = type { int, { uint, ushort, ushort } }
	%struct..RefRect = type { %struct..RefPoint, %struct..RefPoint }

implementation   ; Functions:

uint %BMT_CommitPartDrawObj() {
	%tmp.19111 = getelementptr %struct..RefRect* null, long 0, ubyte 0, ubyte 1, ubyte 2
	%tmp.20311 = getelementptr %struct..RefRect* null, long 0, ubyte 1, ubyte 1, ubyte 2
	ret uint 0
}

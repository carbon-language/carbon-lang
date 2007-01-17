; RUN: llvm-upgrade < %s | llvm-as | opt -basicaa -aa-eval -disable-output

	%struct..RefPoint = type { int, { uint, ushort, ushort } }
	%struct..RefRect = type { %struct..RefPoint, %struct..RefPoint }

implementation   ; Functions:

uint %BMT_CommitPartDrawObj() {
	%tmp.19111 = getelementptr %struct..RefRect* null, long 0, uint 0, uint 1, uint 2
	%tmp.20311 = getelementptr %struct..RefRect* null, long 0, uint 1, uint 1, uint 2
	ret uint 0
}

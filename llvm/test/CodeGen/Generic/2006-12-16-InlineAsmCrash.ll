; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86
; PR1049
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
	%struct.QBasicAtomic = type { int }
	%struct.QByteArray = type { "struct.QByteArray::Data"* }
	"struct.QByteArray::Data" = type { %struct.QBasicAtomic, int, int, sbyte*, [1 x sbyte] }
	%struct.QFactoryLoader = type { %struct.QObject }
	%struct.QImageIOHandler = type { int (...)**, %struct.QImageIOHandlerPrivate* }
	%struct.QImageIOHandlerPrivate = type opaque
	%struct.QImageWriter = type { %struct.QImageWriterPrivate* }
	%struct.QImageWriterPrivate = type { %struct.QByteArray, %struct.QFactoryLoader*, bool, %struct.QImageIOHandler*, int, float, %struct.QString, %struct.QString, uint, %struct.QString, %struct.QImageWriter* }
	"struct.QList<QByteArray>" = type { "struct.QList<QByteArray>::._20" }
	"struct.QList<QByteArray>::._20" = type { %struct.QListData }
	%struct.QListData = type { "struct.QListData::Data"* }
	"struct.QListData::Data" = type { %struct.QBasicAtomic, int, int, int, ubyte, [1 x sbyte*] }
	%struct.QObject = type { int (...)**, %struct.QObjectData* }
	%struct.QObjectData = type { int (...)**, %struct.QObject*, %struct.QObject*, "struct.QList<QByteArray>", ubyte, [3 x ubyte], int, int }
	%struct.QString = type { "struct.QString::Data"* }
	"struct.QString::Data" = type { %struct.QBasicAtomic, int, int, ushort*, ubyte, ubyte, [1 x ushort] }

implementation   ; Functions:

bool %_ZNK12QImageWriter8canWriteEv() {
	%tmp62 = load %struct.QImageWriterPrivate** null		; <%struct.QImageWriterPrivate*> [#uses=1]
	%tmp = getelementptr %struct.QImageWriterPrivate* %tmp62, int 0, uint 9		; <%struct.QString*> [#uses=1]
	%tmp75 = call %struct.QString* %_ZN7QStringaSERKS_( %struct.QString* %tmp, %struct.QString* null )		; <%struct.QString*> [#uses=0]
	call void asm sideeffect "lock\0Adecl $0\0Asetne 1", "=*m"( int* null )
	ret bool false
}

declare %struct.QString* %_ZN7QStringaSERKS_(%struct.QString*, %struct.QString*)

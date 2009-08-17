; RUN: llvm-as < %s | opt -scalarrepl -disable-opt

	%struct.Item = type { [4 x i16], %struct.rule* }
	%struct.rule = type { [4 x i16], i32, i32, i32, %struct.nonterminal*, %struct.pattern*, i8 }
	%struct.nonterminal = type { i8*, i32, i32, i32, %struct.plankMap*, %struct.rule* }
	%struct.plankMap = type { %struct.list*, i32, %struct.stateMap* }
	%struct.list = type { i8*, %struct.list* }
	%struct.stateMap = type { i8*, %struct.plank*, i32, i16* }
	%struct.plank = type { i8*, %struct.list*, i32 }
	%struct.pattern = type { %struct.nonterminal*, %struct.operator*, [2 x %struct.nonterminal*] }
	%struct.operator = type { i8*, i8, i32, i32, i32, i32, %struct.table* }
	%struct.table = type { %struct.operator*, %struct.list*, i16*, [2 x %struct.dimension*], %struct.item_set** }
	%struct.dimension = type { i16*, %struct.Index_Map, %struct.mapping*, i32, %struct.plankMap* }
	%struct.Index_Map = type { i32, %struct.item_set** }
	%struct.item_set = type { i32, i32, %struct.operator*, [2 x %struct.item_set*], %struct.item_set*, i16*, %struct.Item*, %struct.Item* }
	%struct.mapping = type { %struct.list**, i32, i32, i32, %struct.item_set** }

define void @addHP_2_0() {
bb4.i:
	%0 = malloc [0 x %struct.Item]		; <[0 x %struct.Item]*> [#uses=1]
	%.sub.i.c.i = getelementptr [0 x %struct.Item]* %0, i32 0, i32 0		; <%struct.Item*> [#uses=0]
	unreachable
}

; RUN: llvm-as < %s | opt -lower-packed | llvm-dis
@foo = external global <2 x i32>                ; <<2 x i32>*> [#uses=2]
@bar = external global <2 x i32>                ; <<2 x i32>*> [#uses=1]

define void @main() {
        %t0 = load <2 x i32>* @foo              ; <<2 x i32>> [#uses=6]
        %t2 = add <2 x i32> %t0, %t0            ; <<2 x i32>> [#uses=1]
        %t3 = select i1 false, <2 x i32> %t0, <2 x i32> %t2             ; <<2 x i32>> [#uses=1]
        store <2 x i32> %t3, <2 x i32>* @bar
        %c0 = add <2 x i32> < i32 1, i32 1 >, %t0               ; <<2 x i32>> [#uses=0]
        %c1 = add <2 x i32> %t0, zeroinitializer                ; <<2 x i32>> [#uses=0]
        %c2 = select i1 true, <2 x i32> < i32 1, i32 1 >, <2 x i32> %t0         ; <<2 x i32>> [#uses=0]
        store <2 x i32> < i32 4, i32 4 >, <2 x i32>* @foo
        ret void
}


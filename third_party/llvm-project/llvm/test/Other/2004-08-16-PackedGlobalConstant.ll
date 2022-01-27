; RUN: llvm-as < %s | llvm-dis

@foo = global <2 x i32> < i32 0, i32 1 >                ; <<2 x i32>*> [#uses=1]
@bar = external global <2 x i32>                ; <<2 x i32>*> [#uses=1]

define void @main() {
        %t0 = load <2 x i32>, <2 x i32>* @foo              ; <<2 x i32>> [#uses=1]
        store <2 x i32> %t0, <2 x i32>* @bar
        ret void
}


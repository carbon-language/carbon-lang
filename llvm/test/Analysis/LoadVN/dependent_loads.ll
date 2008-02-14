; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | \
; RUN: llvm-dis | not grep sub

%S = type { i32, i8 }

define i8 @test(i8** %P) {
        %A = load i8** %P               ; <i8*> [#uses=1]
        %B = load i8* %A                ; <i8> [#uses=1]
        %X = load i8** %P               ; <i8*> [#uses=1]
        %Y = load i8* %X                ; <i8> [#uses=1]
        %R = sub i8 %B, %Y              ; <i8> [#uses=1]
        ret i8 %R
}

define i8 @test1(%S** %P) {
        %A = load %S** %P               ; <%S*> [#uses=1]
        %B = getelementptr %S* %A, i32 0, i32 1         ; <i8*> [#uses=1]
        %C = load i8* %B                ; <i8> [#uses=1]
        %X = load %S** %P               ; <%S*> [#uses=1]
        %Y = getelementptr %S* %X, i32 0, i32 1         ; <i8*> [#uses=1]
        %Z = load i8* %Y                ; <i8> [#uses=1]
        %R = sub i8 %C, %Z              ; <i8> [#uses=1]
        ret i8 %R
}


; RUN: llvm-as < %s | opt -deadargelim -die | llvm-dis | not grep DEAD

@P = external global i32                ; <i32*> [#uses=1]

; Dead arg only used by dead retval
define internal i32 @test(i32 %DEADARG) {
        ret i32 %DEADARG
}

define internal i32 @test2(i32 %DEADARG) {
        %DEADRETVAL = call i32 @test( i32 %DEADARG )            ; <i32> [#uses=1]
        ret i32 %DEADRETVAL
}

define void @test3(i32 %X) {
        %DEADRETVAL = call i32 @test2( i32 %X )         ; <i32> [#uses=0]
        ret void
}

define internal i32 @foo() {
        %DEAD = load i32* @P            ; <i32> [#uses=1]
        ret i32 %DEAD
}

define internal i32 @id(i32 %X) {
        ret i32 %X
}

define void @test4() {
        %DEAD = call i32 @foo( )                ; <i32> [#uses=1]
        %DEAD2 = call i32 @id( i32 %DEAD )              ; <i32> [#uses=0]
        ret void
}

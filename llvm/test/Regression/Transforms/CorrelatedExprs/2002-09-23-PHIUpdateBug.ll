; RUN: as < %s | opt -cee

implementation

declare void %foo(int)

void %test(int %A, bool %C) {
	br bool %C, label %bb3, label %bb1
bb1:                                    ;[#uses=0]
        %cond212 = setgt int %A, 9              ; <bool> [#uses=1]
        br bool %cond212, label %bb2, label %bb3

bb2:                                    ;[#uses=1]
	%cond = setgt int %A, 7
        br bool %cond, label %bb3, label %bb7

bb3:                                    ;[#uses=1]
	%X = phi int [ 0, %0], [ 12, %bb1]
        call void %foo( int %X )
        br label %bb7

bb7:                                    ;[#uses=2]
        ret void
}


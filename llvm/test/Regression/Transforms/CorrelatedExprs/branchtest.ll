; RUN: echo %s

implementation

declare void %foo(int)

void %test(int %A) {
bb1:                                    ;[#uses=0]
        %cond212 = setgt int %A, 9              ; <bool> [#uses=1]
        br bool %cond212, label %REMOVEbb3, label %bb2

bb2:                                    ;[#uses=1]
        call void %foo( int 123 )
        br label %REMOVEbb3

REMOVEbb3:                                    ;[#uses=2]
        %cond217 = setle int %A, 9            ; <bool> [#uses=1]
        br bool %cond217, label %REMOVEbb5, label %bb4

bb4:                                    ;[#uses=1]
        call void %foo( int 234 )
        br label %REMOVEbb5

REMOVEbb5:                                    ;[#uses=2]
        %cond222 = setgt int %A, 9             ; <bool> [#uses=1]
        br bool %cond222, label %bb7, label %REMOVEbb6

REMOVEbb6:                                    ;[#uses=1]
        call void %foo( int 456 )
        br label %bb7

bb7:                                    ;[#uses=2]
        ret void
}


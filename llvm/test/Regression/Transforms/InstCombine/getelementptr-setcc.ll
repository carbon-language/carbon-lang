; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep getelementptr

bool %test1(short* %P, int %I, int %J) {
    %X = getelementptr short* %P, int %I
    %Y = getelementptr short* %P, int %J

    %C = setlt short* %X, %Y
    ret bool %C
}

bool %test2(short* %P, int %I) {
    %X = getelementptr short* %P, int %I

    %C = setlt short* %X, %P
    ret bool %C
}

int %test3(int* %P, int %A, int %B) {
        %tmp.4 = getelementptr int* %P, int %A          ; <int*> [#uses=1]
        %tmp.9 = getelementptr int* %P, int %B          ; <int*> [#uses=1]
        %tmp.10 = seteq int* %tmp.4, %tmp.9             ; <bool> [#uses=1]
        %tmp.11 = cast bool %tmp.10 to int              ; <int> [#uses=1]
        ret int %tmp.11
}

int %test4(int* %P, int %A, int %B) {
        %tmp.4 = getelementptr int* %P, int %A          ; <int*> [#uses=1]
        %tmp.6 = seteq int* %tmp.4, %P          ; <bool> [#uses=1]
        %tmp.7 = cast bool %tmp.6 to int                ; <int> [#uses=1]
        ret int %tmp.7
}


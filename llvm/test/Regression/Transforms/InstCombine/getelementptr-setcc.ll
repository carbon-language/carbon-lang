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


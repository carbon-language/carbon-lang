; RUN: llvm-as < %s | opt -reassociate -instcombine -constprop -die | llvm-dis | not grep 5

int %test(int %A, int %B) {
        %W = add int %B, -5
        %Y = add int %A, 5
        %Z = add int %W, %Y
        ret int %Z
}

; RUN: if as < %s | opt -reassociate -instcombine -constprop -die | dis | grep 5
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %A, int %B) {
        %W = add int %B, -5
        %Y = add int %A, 5
        %Z = add int %W, %Y
        ret int %Z
}

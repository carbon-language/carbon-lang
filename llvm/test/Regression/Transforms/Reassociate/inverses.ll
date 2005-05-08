; RUN: llvm-as < %s | opt -reassociate -dce | llvm-dis | not grep '\(and\|sub\)'

int %test1(int %a, int %b) {
        %tmp.2 = and int %b, %a
        %tmp.4 = xor int %a, -1
        %tmp.5 = and int %tmp.2, %tmp.4 ; (A&B)&~A == 0
        ret int %tmp.5
}

int %test2(int %a, int %b) {
	%tmp.1 = and int %a, 1234
        %tmp.2 = and int %b, %tmp.1
        %tmp.4 = xor int %a, -1
        %tmp.5 = and int %tmp.2, %tmp.4 ; A&~A == 0
        ret int %tmp.5
}

int %test3(int %b, int %a) {
	%tmp.1 = add int %a, 1234
        %tmp.2 = add int %b, %tmp.1
        %tmp.4 = sub int 0, %a
        %tmp.5 = add int %tmp.2, %tmp.4   ; (b+(a+1234))+-a -> b+1234
        ret int %tmp.5
}

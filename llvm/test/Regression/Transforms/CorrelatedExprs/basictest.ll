; RUN: as < %s | opt -cee -constprop -instcombine -dce | dis | not grep 'REMOVE'

int %test1(int %A) {
	%cond = seteq int %A, 40
	br bool %cond, label %T, label %F
T:
	%REMOVE = add int %A, 2  ; Should become = 42
	ret int %REMOVE
F:
	ret int 8
}

bool %test2(int %A) {
        %cond = seteq int %A, 40
        br bool %cond, label %T, label %F
T:
        %REMOVE = seteq int %A, 2  ; Should become = false
        ret bool %REMOVE
F:
        ret bool false
}

bool %test3(int %A) {
        %cond = setlt int %A, 40
        br bool %cond, label %T, label %F
T:
        %REMOVE = setgt int %A, 47  ; Should become = false
        ret bool %REMOVE
F:
        %REMOVE2 = setge int %A, 40  ; Should become = true
        ret bool %REMOVE2
}

bool %test4(int %A) {
        %cond = setlt int %A, 40
        br bool %cond, label %T, label %F
T:
        %REMOVE = setgt int %A, 47  ; Should become = false
        ret bool %REMOVE
F:
        ret bool false
}

int %test5(int %A, int %B) {
	%cond = setne int %A, %B
	br bool %cond, label %F, label %T
T:
	%C = sub int %A, %B  ; = 0
	ret int %C
F:
	ret int 0
}

bool %test6(int %A) {
        %REMOVE = setlt int %A, 47  ; Should become dead
        %cond = setlt int %A, 40
        br bool %cond, label %T, label %F
T:
        ret bool %REMOVE  ;; == true
F:
        ret bool false
}

bool %test7(int %A) {
	%cond = setlt int %A, 40
	br bool %cond, label %T, label %F
T:
	%REMOVE = xor bool %cond, true
	ret bool %REMOVE
F:
	ret bool false
}

; Test that and expressions are handled...
bool %test8(int %A, int %B) {
	%cond1 = setle int %A, 7
	%cond2 = setle int %B, 7
	%cond = and bool %cond1, %cond2
	br bool %cond, label %T, label %F
T:
	%REMOVE1 = seteq int %A, 9             ; false
	%REMOVE2 = setge int %B, 9             ; false
	%REMOVE = or bool %REMOVE1, %REMOVE2   ; false
	ret bool %REMOVE
F:
	ret bool false
}

; Test that or expressions are handled...
bool %test9(int %A, int %B) {
	%cond1 = setle int %A, 7
	%cond2 = setle int %B, 7
	%cond = or bool %cond1, %cond2
	br bool %cond, label %T, label %F
T:
	ret bool false
F:
	%REMOVE1 = setge int %A, 8             ; true
	%REMOVE2 = setge int %B, 8             ; true
	%REMOVE = or bool %REMOVE1, %REMOVE2   ; true
	ret bool %REMOVE
}

bool %test10(int %A) {
	%cond = setle int %A, 7
	br bool %cond, label %T, label %F
T:
	ret bool false
F:
	%REMOVE = setge int %A, 8
	ret bool %REMOVE
}

; Implement correlated comparisons against non-constants
bool %test11(int %A, int %B) {
	%cond = setlt int %A, %B
	br bool %cond, label %T, label %F
T:
	%REMOVE1 = seteq int %A, %B    ; false
	%REMOVE2 = setle int %A, %B    ; true
	%cond2 = and bool %REMOVE1, %REMOVE2
	ret bool %cond2
F:
	ret bool true
}

bool %test12(int %A) {
	%cond = setlt int %A, 0
	br bool %cond, label %T, label %F
T:
	%REMOVE = setne int %A, 0    ; true
	ret bool %REMOVE
F:
	ret bool false
}

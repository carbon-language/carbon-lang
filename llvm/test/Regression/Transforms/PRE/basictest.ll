; Basic testcases - these are only tested by inspection, but illustrate the 
; basic cases PRE can handle.
;
; RUN: as < %s | opt -pre -disable-output

declare void %use(int)
declare int %get()

void %test0(int %A, int %B) {   ;; Fully redundant
        %X = add int %A, %B
        %Y = add int %A, %B
	call void %use(int %X)
	call void %use(int %Y)
        ret void
}

void %test1(int %cond, int %A, int %B) {
        switch int %cond, label %Out [ 
                int 1, label %B1
                int 2, label %B2
                int 3, label %Cont ]
B1:
	%X1 = add int %A, %B
	call void %use(int %X1)
	br label %Cont
B2:
        %X2 = add int %A, %B
        call void %use(int %X2)
        br label %Cont

Cont:
        br label %Next

Next:
        %X3 = add int %A, %B
        call void %use(int %X3)
        br label %Out

Out:
	ret void
}


void %testloop(bool %cond, int %A, int %B) {
        br label %Loop

Loop:
        %C = add int %A, %B     ; loop invariant
        call void %use(int %C)

        %D = add int %C, %B
        call void %use(int %D)

        br bool %cond, label %Loop, label %Exit
Exit:
        ret void
}



void %test3(bool %cond, int %A, int %B) {
        br bool %cond, label %A, label %B

A:
        %C = add int %A, %B
        call void %use(int %C)
        br label %Merge
B:
        %D = add int %A, %B
        call void %use(int %D)
        br label %Merge

Merge:
        %E = add int %A, %B
        call void %use(int %E)
        ret void
}

void %test4(bool %cond, int %A, int %B) {
        br bool %cond, label %A, label %B

A:
        br label %Merge
B:
        %D = add int %A, %B
        call void %use(int %D)
        br label %Merge

Merge:
        %E = add int %A, %B
        call void %use(int %E)
        ret void
}


int %test5(bool %cond, int %A, int %B) {
        br label %Loop

Loop:
        br bool %cond, label %A, label %B

A:
        br label %Merge
B:
        %D = add int %A, %B
        call void %use(int %D)
        br label %Merge

Merge:
        br bool %cond, label %Loop, label %Out

Out:
        %E = add int %A, %B
        ret int %E
}


void %test6(bool %cond, int %A, int %B) {
        br bool %cond, label %A1, label %Def
A1:     br label %Around
Def:
        %C = add int %A, %B
        call void %use(int %C)
        br bool %cond, label %F1, label %F2
F1:     br label %Around
F2:     br label %Y

Around:
        br label %Y
Y:
        %D = add int %A, %B
        call void %use(int %D)
	ret void
}

void %testloop-load(bool %cond, int* %P, int* %Q) {
        br label %Loop

Loop:
        store int 5, int* %Q          ;; Q may alias P
        %D = load int* %P             ;; Should not hoist load out of loop
        call void %use(int %D)

        br bool %cond, label %Loop, label %Exit
Exit:
        ret void
}

void %test7(bool %cond) {      ;; Test anticipatibility
        br label %Loop

Loop:
        %A = call int %get()
        %B = add int %A, %A          ; Cannot hoist from loop
        call void %use(int %B)

        br bool %cond, label %Loop, label %Exit
Exit:
        ret void
}


void %test8(bool %cond, int %A, int %B) {   ;; Test irreducible loop
        br bool %cond, label %LoopHead, label %LoopMiddle

LoopHead:
        %C = add int %A, %B          ; Should hoist from loop
        call void %use(int %C)
        br label %LoopMiddle

LoopMiddle:

        br bool %cond, label %LoopHead, label %Exit
Exit:
        %D = add int %A, %B
        call void %use(int %D)
        ret void
}


void %test9(bool %cond, int %A, int %B) {   ;; Test irreducible loop
        br bool %cond, label %LoopHead, label %LoopMiddle

LoopHead:
        call int %get()              ; random function call
        br label %LoopMiddle

LoopMiddle:
        %C = add int %A, %B          ; Should hoist from loop
        call void %use(int %C)
        br bool %cond, label %LoopHead, label %Exit
Exit:
        ret void
}

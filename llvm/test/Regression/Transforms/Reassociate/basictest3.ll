; RUN: llvm-as < %s | opt -reassociate -gcse | llvm-dis | grep add | wc -l | grep 6
; Each of these functions should turn into two adds each.

%e = external global int
%a = external global int
%b = external global int
%c = external global int
%f = external global int

implementation

void %test1() {
        %A = load int* %a
        %B = load int* %b
        %C = load int* %c
        %t1 = add int %A, %B
	%t2 = add int %t1, %C
        %t3 = add int %C, %A
	%t4 = add int %t3, %B
        store int %t2, int* %e  ; e = (a+b)+c;
        store int %t4, int* %f  ; f = (a+c)+b
        ret void
}

void %test2() {
        %A = load int* %a
        %B = load int* %b
        %C = load int* %c
	%t1 = add int %A, %B
	%t2 = add int %t1, %C
	%t3 = add int %C, %A
	%t4 = add int %t3, %B
        store int %t2, int* %e  ; e = c+(a+b)
        store int %t4, int* %f  ; f = (c+a)+b
        ret void
}

void %test3() {
        %A = load int* %a
        %B = load int* %b
        %C = load int* %c
	%t1 = add int %B, %A
	%t2 = add int %t1, %C
	%t3 = add int %C, %A
	%t4 = add int %t3, %B
        store int %t2, int* %e  ; e = c+(b+a)
        store int %t4, int* %f  ; f = (c+a)+b
        ret void
}


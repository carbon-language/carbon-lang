%A = global int 0

int %main() {
	%Ret = call int %test(bool true, int 0)	
	ret int %Ret
}

int %test(bool %c, int %A) {
	br bool %c, label %Taken1, label %NotTaken

Cont:
	%V = phi int [0, %NotTaken], 
	              [ sub (int cast (int* %A to int), int 1234), %Taken1]
	ret int 0

NotTaken:
	br label %Cont	

Taken1:
	%B = seteq int %A, 0
	; Code got inserted here, breaking the condition code.
	br bool %B, label %Cont, label %ExitError

ExitError:
	ret int 12

}

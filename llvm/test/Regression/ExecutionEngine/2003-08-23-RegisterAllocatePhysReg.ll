; This testcase exposes a bug in the local register allocator where it runs out
; of registers (due to too many overlapping live ranges), but then attempts to
; use the ESP register (which is not allocatable) to hold a value.

int %main(uint %A) {
	%Ap2 = alloca uint, uint %A   ; ESP gets used again...
	%B = add uint %A, 1 	      ; Produce lots of overlapping live ranges
	%C = add uint %A, 2
	%D = add uint %A, 3
	%E = add uint %A, 4
	%F = add uint %A, 5
	%G = add uint %A, 6
	%H = add uint %A, 7
	%I = add uint %A, 8
	%J = add uint %A, 9
	%K = add uint %A, 10

	store uint %A, uint *%Ap2      ; Uses of all of the values
	store uint %B, uint *%Ap2
	store uint %C, uint *%Ap2
	store uint %D, uint *%Ap2
	store uint %E, uint *%Ap2
	store uint %F, uint *%Ap2
	store uint %G, uint *%Ap2
	store uint %H, uint *%Ap2
	store uint %I, uint *%Ap2
	store uint %J, uint *%Ap2
	store uint %K, uint *%Ap2
	ret int 0
}

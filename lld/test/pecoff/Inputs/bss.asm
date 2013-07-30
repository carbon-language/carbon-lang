.586
.model flat, c

extern ExitProcess@4 : PROC

_BSS	SEGMENT
	_x	DD	064H DUP (?)
	_y	DD	064H DUP (?)
_BSS	ENDS

.code
start:
	mov eax, 42
	mov _x, eax
	mov eax, _x
	push eax
	call ExitProcess@4
end start

end

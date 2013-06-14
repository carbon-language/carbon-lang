;;; ml hello.asm /link /subsystem:windows /defaultlib:kernel32.lib \
;;;    /defaultlib:user32.lib /out:hello.exe /entry:main

.386
.model flat, c

extern MessageBoxA@16 : PROC
extern ExitProcess@4 : PROC

.data
	caption db "Hello", 0
	message db "Hello World", 0

.code
main:
	mov eax, 0
	push eax
	push offset caption
	push offset message
	push eax
	call MessageBoxA@16
	push eax
	call ExitProcess@4
end main

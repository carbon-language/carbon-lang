implementation

; Test "stripped" format where nothing is symbolic... this is how the bytecode
; format looks anyways (except for negative vs positive offsets)...
;
void "void"(int, int)   ; Def %0, %1
	const int 0          ; Def 2
	const int -4         ; Def 3
begin
	br label %1

	add int %0, %1    ; Def 4
	sub int %4, %3    ; Def 5
	setle int %5, %2  ; Def 0 - bool plane
	br bool %0, label %2, label %1

	add int %0, %1    ; Def 6
	sub int %4, %3    ; Def 7
	setle int %7, %2  ; Def 1 - bool plane
	ret void
end

; This function always returns zero
int "zarro"()
	const uint 4000000000        ; Def 0 - uint plane
	const int 0                  ; Def 0 - int plane
begin
Startup:
	ret int %0
end


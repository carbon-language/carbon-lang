implementation

void "test function"(int %i0, int %j0)
	%x = int 1
begin
Startup:                               ; Basic block #0
	%i1 = add int %i0, 1
	%j1 = add int %j0, 1
	%x = setle int %i1, %j1
	br bool %x, label %Increment, label %Decrement

Merge:                                 ; Basic block #3
	%i4 = phi int [%i2, %Increment], [%i3,%Decrement] ; Forward ref vars...
	%j2 = add int %j1, %i4
	ret void

Increment:                             ; Basic block #1
	%i2 = add int %i1, 1
	br label %Merge

Decrement:                             ; Basic block #2
	%i3 = sub int %i1, %x
	br label %Merge
end


; Test "stripped" format where nothing is symbolic... this is how the bytecode
; format looks anyways (except for negative vs positive offsets)...
;
void "void"(int, int)   ; Def %0, %1
	int 0          ; Def 2
	int -4         ; Def 3
begin
	add int %0, %1    ; Def 4
	sub int %4, %3    ; Def 5
	setle int %5, %2  ; Def 0 - bool plane
	br bool %0, label %1, label %0

	add int %0, %1    ; Def 6
	sub int %4, %3    ; Def 7
	setle int %7, %2  ; Def 1 - bool plane
	ret void
end

; This function always returns zero
int "zarro"()
	uint 4000000000        ; Def 0 - uint plane
	int 0                  ; Def 0 - int plane
begin
Startup:
	ret int %0
end


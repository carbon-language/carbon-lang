  const long 12

implementation

ulong "const removal"() 
	const long 12345
	%q = const uint 4000000000   ; Def %q - uint plane
	const ulong 123              ; Def 0 - ulong plane
	const ulong 4000000000000    ; Def 1 - ulong plane
begin
        ret ulong %1
end

void "dce #1"() 
        const int 0
begin
        ret void

	ret void                    ; Unreachable label
end

void "basic block merge #1"(int %a, uint %b, bool %c, ulong %d) 
begin
        add int %0, %0
        br label %TestName         ; Can be merged with next block
TestName:
        add uint %0, %0
        br label %1                 ; Can be merged with next block
        add ulong %0, %0
        ret void
end

void "const prop #1"()
   %x = const int 0                  ; Def %x - int plane
        const int 0                  ; Def 0 - int plane
        const bool false
begin
Startup:
        %x = seteq int %0, %x
        br bool %x, label %0, label %Startup  ; Can be eliminated by const prop

	ret void
end

int "const prop #2"()
begin
	%x = add int 1, 1            ; Instrs can be const prop'd away
        %y = sub int -1, 1
        %z = add int %x, %y
	ret int %z                     ; Should equal %0
end

sbyte "const prop #3"()              ; Instrs can be const prop'd away
begin
	%x = add sbyte 127, 127      ; Must wrap around correctly!
        %y = sub sbyte 1, -1
        %z = add sbyte %x, %y
	ret sbyte %z                   ; Should equal %0!
end

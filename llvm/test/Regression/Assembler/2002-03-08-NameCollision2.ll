; Another name collision problem.  Here the problem was that if a forward 
; declaration for a method was found, that this would cause spurious conflicts
; to be detected between locals and globals.
;
%Var = uninitialized global int

declare void "foo"()

implementation

void "foo"()
begin
	%Var = alloca int  ; Conflict with global var
	ret void
end


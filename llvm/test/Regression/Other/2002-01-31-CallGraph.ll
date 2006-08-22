;  Call graph construction crash: Not handling indirect calls right
;
; RUN: opt -analyze -callgraph %s
;

%FunTy = type int(int)

implementation

void "invoke"(%FunTy *%x)
begin
	%foo = call %FunTy* %x(int 123)
	ret void
end


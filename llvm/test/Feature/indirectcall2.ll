implementation

ulong "test"(ulong %X)
begin
	ret ulong %X
end

ulong "fib"(ulong %n)
begin
  %T = setlt ulong %n, 2       ; {bool}:0
  br bool %T, label %BaseCase, label %RecurseCase

RecurseCase:
  %result = call ulong %test(ulong %n)
  br label %BaseCase

BaseCase:
  %X = phi ulong [1, %0], [2, %RecurseCase]
  ret ulong %X
end


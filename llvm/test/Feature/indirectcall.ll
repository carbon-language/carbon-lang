implementation

declare int "atoi"(sbyte *)

ulong "fib"(ulong %n)
begin
  setlt ulong %n, 2       ; {bool}:0
  br bool %0, label %BaseCase, label %RecurseCase

BaseCase:
  ret ulong 1

RecurseCase:
  %n2 = sub ulong %n, 2
  %n1 = sub ulong %n, 1
  %f2 = call ulong(ulong) * %fib(ulong %n2)
  %f1 = call ulong(ulong) * %fib(ulong %n1)
  %result = add ulong %f2, %f1
  ret ulong %result
end

ulong "realmain"(int %argc, sbyte ** %argv)
begin
  seteq int %argc, 2      ; {bool}:0
  br bool %0, label %HasArg, label %Continue
HasArg:
  ; %n1 = atoi(argv[1])
  %n1 = add int 1, 1
  br label %Continue

Continue:
  %n = phi int [%n1, %HasArg], [1, %0]
  %N = cast int %n to ulong
  %F = call ulong(ulong) *%fib(ulong %N)
  ret ulong %F
end

ulong "trampoline"(ulong %n, ulong(ulong)* %fibfunc)
begin
  %F = call ulong(ulong) *%fibfunc(ulong %n)
  ret ulong %F
end

int "main"()
begin
  %Result = call ulong %trampoline(ulong 10, ulong(ulong) *%fib)
  %Result = cast ulong %Result to int
  ret int %Result
end


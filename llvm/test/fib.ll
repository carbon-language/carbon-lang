
;; TODO:
;; [ ] Get rid out outside class & begin stuff
;; [ ] Allow global const pool to be expanded continually
;; [ ] Support global variable declaration & definition
;; [ ] Support function definition:  %fib = prototype ulong (ulong)
;; [x] Support Type definition

implementation

ulong "fib"(ulong %n)
begin
  setlt ulong %n, 2       ; {bool}:0
  br bool %0, label %BaseCase, label %RecurseCase

BaseCase:
  ret ulong 1

RecurseCase:
  %n2 = sub ulong %n, 2
  %n1 = sub ulong %n, 1
  %f2 = call ulong(ulong) %fib(ulong %n2)
  %f1 = call ulong(ulong) %fib(ulong %n1)
  %result = add ulong %f2, %f1
  ret ulong %result
end

ulong "main"(int %argc, sbyte ** %argv)
;;  %n2 = int 1
begin
  seteq int %argc, 2      ; {bool}:0
  br bool %0, label %HasArg, label %Continue
HasArg:
  ; %n1 = atoi(argv[1])
;;;  %n1 = add int 1, 1
  br label %Continue

Continue:
;;;  %n = phi int %n1, %n2
  %N = add ulong 1, 1       ;; TODO: CAST
  %F = call ulong(ulong) %fib(ulong %N)
  ret ulong %F
end

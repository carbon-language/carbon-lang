; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

implementation

declare int "printf"(sbyte*, ...)   ;; Prototype for: int __builtin_printf(const char*, ...)

int "testvarar"()
begin
	call int(sbyte*, ...) *%printf(sbyte * null, int 12, sbyte 42);
	ret int %0
end



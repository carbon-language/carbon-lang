implementation

declare int "printf"(sbyte*, ...)   ;; Prototype for: int __builtin_printf(const char*, ...)

int "testvarar"()
begin
	cast int 0 to sbyte*
	call int(sbyte*, ...) %printf(sbyte * %0, int 12, sbyte 42);
	ret int %0
end



implementation

declare int "printf"(sbyte*, ...)   ;; Prototype for: int __builtin_printf(const char*, ...)

int "testvarar"()
begin
	call int(sbyte*, ...) *%printf(sbyte * null, int 12, sbyte 42);
	ret int %0
end



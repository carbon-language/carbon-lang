implementation

int "FuncToInline"()
begin
	%x = add int 1, 1            ; Instrs can be const prop'd away
        %y = sub int -1, 1
        %z = add int %x, %y
	ret int %z                     ; Should equal %0
end

int "FuncToInlineInto"(int %arg)     ; Instrs can be const prop'd away
begin
	%x = add int %arg, 1
        %y = sub int 1, -1
        %p = call int() %FuncToInline()
        %z = add int %x, %y
        %q = add int %p, %z

	ret int %q
end

int "main"()
begin
        %z = call int(int) %FuncToInlineInto(int 1)
        ret int %z
end


implementation

; Bytecode gets a constant pool block, that constains:
; type   plane: int(int)

int "main"(int %argc)   ; TODO: , sbyte **argv, sbyte **envp)
begin
        %retval = call int (int) *%test(int %argc)
        %two    = add int %retval, %retval
        ret int %two
end

int "test"(int %i0)
begin
    ret int %i0
end

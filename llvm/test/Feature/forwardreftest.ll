  %myty = type int 
  %myfn = type float (int,double,uint,short)
  type int(%myfn)
  type int(int)
  type int(int(int))
implementation

; This function always returns zero
int "zarro"(int %Func)
	%q = const uint 4000000000
	%p = const int 0
begin
Startup:
    add int %p, 10
    ret int %p
end

int "test"(int) 
    %thisfuncty = type int (int) *
begin
    add %thisfuncty %zarro, %test
    add %thisfuncty %test, %foozball
    ret int 0
end

int "foozball"(int)
begin
    ret int 0
end


  %myty = type int 
  %myfn = type float (int,double,uint,short)
  type int(%myfn)
  type int(int)
  type int(int(int))

  %thisfuncty = type int (int) *
implementation

; This function always returns zero
int "zarro"(int %Func)
begin
Startup:
    add int 0, 10
    ret int 0 
end

int "test"(int) 
begin
    add %thisfuncty %zarro, %test
    add %thisfuncty %test, %foozball
    ret int 0
end

int "foozball"(int)
begin
    ret int 0
end


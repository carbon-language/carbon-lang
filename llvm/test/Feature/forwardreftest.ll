  %myty = type int 
  %myfn = type float (int,double,uint,short)
  type int(%myfn)
  type int(int)
  type int(int(int))

  %thisfuncty = type int (int) *
implementation

declare void %F(%thisfuncty, %thisfuncty, %thisfuncty)

; This function always returns zero
int %zarro(int %Func)
begin
Startup:
    add int 0, 10
    ret int 0 
end

int %test(int) 
begin
    call void %F(%thisfuncty %zarro, %thisfuncty %test, %thisfuncty %foozball)
    ret int 0
end

int %foozball(int)
begin
    ret int 0
end


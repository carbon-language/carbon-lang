%struct = type { int , {float, {ubyte } } , ulong }
%complexty = type {int, {[4 x sbyte *], float}, double}

implementation

int "main"()
begin
  call int %testfunction(uint 0, uint 1)
  ret int 0
end

int "testfunction"(uint %i0, uint %j0)
begin
    %array0 = malloc [4 x ubyte]            ; yields {[4 x ubyte]*}:array0
    %size   = add uint 2, 2                 ; yields {uint}:size = uint %4
    %array1 = malloc ubyte, uint 4          ; yields {ubyte*}:array1
    %array2 = malloc ubyte, uint %size      ; yields {ubyte*}:array2

    store ubyte 123, [4 x ubyte]* %array0, uint 0, uint 2
    free [4x ubyte]* %array0
    free ubyte* %array1
    free ubyte* %array2


    %aa = alloca %complexty, uint 5
    store sbyte *null, %complexty* %aa, uint %i0, ubyte 1, ubyte 0, uint %j0
    
    %ptr = alloca int                       ; yields {int*}:ptr
    store int 3, int* %ptr                  ; yields {void}
    %val = load int* %ptr                   ; yields {int}:val = int %3

    %sptr = alloca %struct                  ; yields {%struct*}:sptr
    %ubsptr = getelementptr %struct * %sptr, uint 0, ubyte 1, ubyte 1  ; yields {{ubyte}*}:ubsptr
    store ubyte 4, {ubyte} * %ubsptr, uint 0, ubyte 0

    ret int 3
end


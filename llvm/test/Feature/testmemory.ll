%struct = type { int , {float, {ubyte } } , ulong }
%complexty = type {int, {[4 x sbyte *], float}, double}

implementation

int "main"()
begin
  call int %testfunction(long 0, long 1)
  ret int 0
end

int "testfunction"(long %i0, long %j0)
begin
    %array0 = malloc [4 x ubyte]            ; yields {[4 x ubyte]*}:array0
    %size   = add uint 2, 2                 ; yields {uint}:size = uint %4
    %array1 = malloc ubyte, uint 4          ; yields {ubyte*}:array1
    %array2 = malloc ubyte, uint %size      ; yields {ubyte*}:array2

    %idx = getelementptr [4 x ubyte]* %array0, long 0, long 2
    store ubyte 123, ubyte* %idx
    free [4x ubyte]* %array0
    free ubyte* %array1
    free ubyte* %array2


    %aa = alloca %complexty, uint 5
    %idx2 = getelementptr %complexty* %aa, long %i0, ubyte 1, ubyte 0, long %j0
    store sbyte *null, sbyte** %idx2
    
    %ptr = alloca int                       ; yields {int*}:ptr
    store int 3, int* %ptr                  ; yields {void}
    %val = load int* %ptr                   ; yields {int}:val = int %3

    %sptr = alloca %struct                  ; yields {%struct*}:sptr
    %ubsptr = getelementptr %struct * %sptr, long 0, ubyte 1, ubyte 1  ; yields {{ubyte}*}:ubsptr
    %idx3 = getelementptr {ubyte} * %ubsptr, long 0, ubyte 0
    store ubyte 4, ubyte* %idx3

    ret int 3
end


  %ptrty = type int *

implementation

[[2 x int]] "test function"(int %i0, int %j0)
        %array = [[2 x int]] [
                   [2 x int] [ int 12, int 52 ]
                 ]
begin
	ret [[2x int]] %array
end


implementation

void "foo"(int * %x, int * %y)
begin
; <label>:0					;[#uses=1]
	br label %Top

Top:					;[#uses=4]
	phi int [ 0, %0 ], [ 1, %Top ], [ 2, %Next ]		; <int>:0	[#uses=0]
	br bool true, label %Top, label %Next

Next:					;[#uses=2]
	br label %Top
end

  %int = type int

implementation

int "squared"(%int %i0)
begin
	switch int %i0, label %Default [ 
		int 1, label %Case1
		int 2, label %Case2
		int 4, label %Case4 ]

Default:
    ret int -1                      ; Unrecognized input value

Case1:
    ret int 1
Case2:
    ret int 4
Case4:
    ret int 16
end

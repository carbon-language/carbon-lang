implementation

;; Test live variable analysis:
;; -- phi argument is also used as first class value

int "PhiTest"(int %i, int %j)
begin
Start:
	%i1 = add int %i, %j
	br label %L1Header

L1Header:
	%i2 = phi int [%i1, %Start], [%i4, %L1Header]

	%i3 = add int %i1, 0
	%i4 = add int %i2, %i3
	%L1Done = setgt int %i4, 10
	br bool %L1Done, label %L1Done, label %L1Header

L1Done:
	ret int %i4
end


int "main"()
begin
bb0:
	%result = call int %PhiTest( int 9, int 17 )
	ret int %result
end

implementation

int "looptest"(int %i, int %j)
begin
	%whichLoop = setlt int %i, %j
	br bool %whichLoop, label %Loop1Header, label %Loop2Header

Loop1Header:
	%i1 = add int 0, 0             ; %i1 = 0
	br label %L1Top
L1Top:
	%i2 = phi int [%i1, %Loop1Header], [%i3, %L1Body]
	%L1Done = seteq int %i2, %j
	br bool %L1Done, label %L1End, label %L1Body
L1Body:
	%i3 = add int %i2, 2
	br label %L1Top
L1End:
	%v0 = add int %i2, %j         ; %v0 = 3 * %j
	br label %Merge

Loop2Header:
	%m1 = add int 0, 0
	%k1 = add int 0, %i
	br label %L2Top
L2Top:
	%k2 = phi int [%k1, %Loop2Header], [%k3, %L2Body]
	%m2 = phi int [%m1, %Loop2Header], [%m3, %L2Body]
	%L2Done = seteq int %k2, 0
	br bool %L2Done, label %L2End, label %L2Body
L2Body:
	%k3 = sub int %k2, 1
	%m3 = add int %m2, %j
	br label %L2Top
L2End:
	%v1 = add int %m2, %k2
	br label %Merge

Merge:
	%v2 = phi int [%v0, %L1End], [%v1, %L2End]
	ret int %v2
end

int "main"()
begin
	call int %looptest(int 0, int 12)
	ret int %0
end


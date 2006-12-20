; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%somestr = constant [11x sbyte] c"hello world"
%array   = constant [2 x int] [ int 12, int 52 ]
           constant { int, int } { int 4, int 3 }

implementation
 
[2 x int]* %testfunction(int %i0, int %j0)
begin
	ret [2x int]* %array
end

sbyte* %otherfunc(int, double)
begin
	%somestr = getelementptr [11x sbyte]* %somestr, long 0, long 0
	ret sbyte* %somestr
end

sbyte* %yetanotherfunc(int, double)
begin
	ret sbyte* null            ; Test null
end

uint %negativeUnsigned() {
        ret uint -1
}

int %largeSigned() {
       ret int 3900000000
}

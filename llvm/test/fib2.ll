%__intern_LC0 = constant [sbyte] c"fib returned: "
%__intern_LC1 = constant [sbyte] c"\0A\00"

implementation

declare void "__main"()

declare int "atoi"(sbyte *)

declare void "printVal"([sbyte] *)
declare void "printVal"(uint)

int "main"(int %argc, sbyte * * %argv)
begin
bb1:					;[#uses=1]
	call void () * %__main( )
	%cond1003 = setne int %argc, 2		; <bool>	[#uses=1]
	br bool %cond1003, label %bb4, label %bb2

bb2:					;[#uses=2]
	%cast1006 = cast ulong 8 to sbyte * *		; <sbyte * *>	[#uses=1]
	%reg1004 = add sbyte * * %argv, %cast1006		; <sbyte * *>	[#uses=1]
	%reg110 = load sbyte * * %reg1004		; <sbyte *>	[#uses=1]
	%reg109 = call int (sbyte *) * %atoi( sbyte * %reg110 )		; <int>	[#uses=1]
	br label %bb4

bb4:					;[#uses=3]
	%reg126 = phi int [ %reg109, %bb2 ], [ 15, %bb1 ]		; <int>	[#uses=3]
	%cast1007 = cast int %reg126 to uint		; <uint>	[#uses=1]
	%cond1005 = setgt uint %cast1007, 1		; <bool>	[#uses=1]
	br bool %cond1005, label %bb6, label %bb7

bb6:					;[#uses=2]
	%reg115 = add int %reg126, -2		; <int>	[#uses=1]
	%cast1008 = cast int %reg115 to uint		; <uint>	[#uses=1]
	%reg116 = call uint (uint) * %fib( uint %cast1008 )		; <uint>	[#uses=1]
	%reg118 = add int %reg126, -1		; <int>	[#uses=1]
	%cast1009 = cast int %reg118 to uint		; <uint>	[#uses=1]
	%reg119 = call uint (uint) * %fib( uint %cast1009 )		; <uint>	[#uses=1]
	%reg127 = add uint %reg116, %reg119		; <uint>	[#uses=1]
	br label %bb7

bb7:					;[#uses=2]
	%reg128 = phi uint [ %reg127, %bb6 ], [ 1, %bb4 ]		; <uint>	[#uses=1]
	call void %printVal([sbyte] * %__intern_LC0)
	call void %printVal(uint %reg128 )
	call void %printVal([sbyte] * %__intern_LC1)
	ret int 0
end

uint "fib"(uint %n)
begin
bb1:					;[#uses=0]
	%cond1000 = setgt uint %n, 1		; <bool>	[#uses=1]
	br bool %cond1000, label %bb3, label %bb2

bb2:					;[#uses=1]
	ret uint 1

bb3:					;[#uses=1]
	%cast1001 = cast long -2 to uint		; <uint>	[#uses=1]
	%reg112 = add uint %n, %cast1001		; <uint>	[#uses=1]
	%reg113 = call uint (uint) * %fib( uint %reg112 )		; <uint>	[#uses=1]
	%cast1002 = cast long -1 to uint		; <uint>	[#uses=1]
	%reg115 = add uint %n, %cast1002		; <uint>	[#uses=1]
	%reg116 = call uint (uint) * %fib( uint %reg115 )		; <uint>	[#uses=1]
	%reg110 = add uint %reg113, %reg116		; <uint>	[#uses=1]
	ret uint %reg110
end

%__intern_LC0 = global [sbyte] c"Ack(3, \00"
%__intern_LC1 = global [sbyte] c") = \00"
%__intern_LC2 = global [sbyte] c"\0A\00"		; <[15 x sbyte] *>	[#uses=1]
implementation

declare void "__main"()

declare void "printVal"([sbyte] *)
declare void "printVal"(int)

declare int "atoi"(sbyte *)

int "main"(int %argc, sbyte * * %argv)
begin
bb1:					;[#uses=1]
	call void () * %__main( )
	%cond1002 = setne int %argc, 2		; <bool>	[#uses=1]
	br bool %cond1002, label %bb4, label %bb2

bb2:					;[#uses=2]
	%cast1010 = cast ulong 8 to sbyte * *		; <sbyte * *>	[#uses=1]
	%reg1003 = add sbyte * * %argv, %cast1010		; <sbyte * *>	[#uses=1]
	%reg110 = load sbyte * * %reg1003		; <sbyte *>	[#uses=1]
	%reg109 = call int (sbyte *) * %atoi( sbyte * %reg110 )		; <int>	[#uses=1]
	br label %bb4

bb4:					;[#uses=2]
	%reg132 = phi int [ %reg109, %bb2 ], [ 5, %bb1 ]		; <int>	[#uses=4]
	%cond1004 = setne ulong 3, 0		; <bool>	[#uses=1]
	br bool %cond1004, label %bb6, label %bb5

bb5:					;[#uses=2]
	%reg115 = add int %reg132, 1		; <int>	[#uses=1]
	br label %bb9

bb6:					;[#uses=1]
	%cond1005 = setne int %reg132, 0		; <bool>	[#uses=1]
	br bool %cond1005, label %bb8, label %bb7

bb7:					;[#uses=2]
	%cast1006 = cast ulong 1 to int		; <int>	[#uses=1]
	%cast1007 = cast ulong 2 to int		; <int>	[#uses=1]
	%reg119 = call int (int, int) * %Ack( int %cast1007, int %cast1006 )		; <int>	[#uses=1]
	br label %bb9

bb8:					;[#uses=2]
	%reg121 = add int %reg132, -1		; <int>	[#uses=1]
	%cast1008 = cast ulong 3 to int		; <int>	[#uses=1]
	%reg122 = call int (int, int) * %Ack( int %cast1008, int %reg121 )		; <int>	[#uses=1]
	%cast1009 = cast ulong 2 to int		; <int>	[#uses=1]
	%reg124 = call int (int, int) * %Ack( int %cast1009, int %reg122 )		; <int>	[#uses=1]
	br label %bb9

bb9:					;[#uses=3]
	%reg135 = phi int [ %reg124, %bb8 ], [ %reg119, %bb7 ], [ %reg115, %bb5 ]		; <int>	[#uses=1]
	call void %printVal([sbyte] *%__intern_LC0)
	call void %printVal(int %reg132)
	call void %printVal([sbyte] *%__intern_LC1)
	call void %printVal(int %reg135)
	call void %printVal([sbyte] *%__intern_LC2)
	ret int 0
end

int "Ack"(int %M, int %N)
begin
bb1:					;[#uses=2]
	br label %bb2

bb2:					;[#uses=3]
	%reg121 = phi int [ %reg117, %bb6 ], [ 1, %bb5 ], [ %N, %bb1 ]		; <int>	[#uses=3]
	%reg122 = phi int [ %reg115, %bb6 ], [ %reg123, %bb5 ], [ %M, %bb1 ]		; <int>	[#uses=4]
	%cond1000 = setne int %reg122, 0		; <bool>	[#uses=1]
	br bool %cond1000, label %bb4, label %bb3

bb3:					;[#uses=1]
	%reg109 = add int %reg121, 1		; <int>	[#uses=1]
	ret int %reg109

bb4:					;[#uses=1]
	%cond1001 = setne int %reg121, 0		; <bool>	[#uses=1]
	br bool %cond1001, label %bb6, label %bb5

bb5:					;[#uses=3]
	%reg123 = add int %reg122, -1		; <int>	[#uses=1]
	br label %bb2

bb6:					;[#uses=3]
	%reg115 = add int %reg122, -1		; <int>	[#uses=1]
	%reg116 = add int %reg121, -1		; <int>	[#uses=1]
	%reg117 = call int (int, int) * %Ack( int %reg122, int %reg116 )		; <int>	[#uses=1]
	br label %bb2
end

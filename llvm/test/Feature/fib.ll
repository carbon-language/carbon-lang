implementation

uint "fib"(uint %n)
begin
bb1:                                    ;[#uses=0]
        %cond1000 = setgt uint %n, 1            ; <bool> [#uses=1]
        br bool %cond1000, label %bb3, label %bb2

bb2:                                    ;[#uses=1]
        ret uint 1

bb3:                                    ;[#uses=1]
        %reg112 = sub uint %n, 2			; <uint> [#uses=1]
        %reg113 = call uint %fib( uint %reg112 )        ; <uint> [#uses=1]
        %reg115 = sub uint %n, 1			; <uint> [#uses=1]
        %reg116 = call uint %fib( uint %reg115 )        ; <uint> [#uses=1]
        %reg110 = add uint %reg113, %reg116             ; <uint> [#uses=1]
        ret uint %reg110
end

int "main"(int %argc, sbyte * * %argv)
begin
bb1:                                    ;[#uses=0]
        %reg109 = call uint %fib( uint 6 )              ; <uint> [#uses=1]
        %cast1004 = cast uint %reg109 to int            ; <int> [#uses=1]
        ret int %cast1004
end


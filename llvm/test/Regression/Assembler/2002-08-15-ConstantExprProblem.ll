%.LC0 = internal global [12 x sbyte] c"hello world\00"

implementation   ; Functions:

sbyte* %test() {
        br label %BB1

BB1:                                    ;[#uses=2]
        %ret = phi sbyte* [ getelementptr ([12 x sbyte]* %.LC0, long 0, long 0), %0 ], [ null, %BB2 ]
        ret sbyte* %ret

BB2:
        br label %BB1
}


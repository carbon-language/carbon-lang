%.LC0 = internal global [12 x sbyte] c"hello world\00"          ; <[12 x sbyte]*> [#uses=1]

implementation   ; Functions:

sbyte* %test() {
        ret sbyte* getelementptr ([12 x sbyte]* %.LC0, long 0, long 0)
}


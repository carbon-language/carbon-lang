%.LC0 = internal global [12 x sbyte] c"Hello World\00"

implementation

declare int %puts(sbyte*)

int %main() {
        %reg210 = call int %puts( sbyte* getelementptr ([12 x sbyte]* %.LC0, long 0, long 0) )
        ret int 0
}


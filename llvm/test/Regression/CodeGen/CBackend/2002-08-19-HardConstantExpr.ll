%array = constant [2 x int] [ int 12, int 52 ]          ; <[2 x int]*> [#uses=1]
%arrayPtr = global int* getelementptr ([2 x int]* %array, long 0, long 0)               ; <int**> [#uses=1]


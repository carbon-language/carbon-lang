%array = constant [2 x int] [ int 12, int 52 ]          ; <[2 x int]*> [#uses=1]
%arrayPtr = global int* getelementptr ([2 x int]* %array, uint 0, uint 0)               ; <int**> [#uses=1]
%arrayPtr5 = global int* getelementptr (int** %arrayPtr, uint 0, uint 5)                ; <int**> [#uses=0]


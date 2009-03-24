// RUN: clang-cc -E %s | grep 'FUNC (3 +1);'

#define F(a) a 
#define FUNC(a) (a+1) 

F(FUNC) FUNC (3); /* final token sequence is FUNC(3+1) */ 


// RUN: clang-cc -E %s | grep -F '"f(1, 2)" "g((x=y++, y))"' &&
// RUN: clang-cc -E %s | grep -F '"{a=1" "b=2;}"' &&
// RUN: clang-cc -E %s | grep -F '"<" "["' &&
// RUN: clang-cc -E %s | grep -F '"(,)" "(...)"' &&
// RUN: clang-cc -E %s | grep -F '{a=1 c=3; b=2;}' &&
// RUN: clang-cc -E %s | grep -F '"a COMMA b" "(a, b)"'

#define M(x, y) #x #y

M( f(1, 2), g((x=y++, y))) 
M( {a=1 , b=2;} ) /* A semicolon is not a comma */ 
M( <, [ ) /* Passes the arguments < and [ */ 
M( (,), (...) ) /* Passes the arguments (,) and (...) */ 

#define START_END(start, end) start c=3; end 

START_END( {a=1 , b=2;} ) /* braces are not parentheses */ 

/* 
 * To pass a comma token as an argument it is 
 * necessary to write: 
 */ 
#define COMMA , 
 
M(a COMMA b, (a, b)) 


// RUN: clang-cc -E %s | grep 'A B C A B A C A B C A'

#define A A B C 
#define B B C A 
#define C C A B 
 
A 


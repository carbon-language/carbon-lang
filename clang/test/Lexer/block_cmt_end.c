/*
RUN: clang -E %s | grep bar &&
RUN: clang -E %s | grep foo &&
RUN: clang -E %s | not grep abc &&
RUN: clang -E %s | not grep xyz
*/

/* abc

ends with normal escaped newline:
*\  
/

bar

/* xyz


ends with a trigraph escaped newline:
*??/    
/

foo


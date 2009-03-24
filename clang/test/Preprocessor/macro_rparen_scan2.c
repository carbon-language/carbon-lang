// RUN: clang-cc -E %s | grep -F 'static int glob = (1 + 1 );'

#define R_PAREN ) 

#define FUNC(a) a 

static int glob = (1 + FUNC(1 R_PAREN ); 


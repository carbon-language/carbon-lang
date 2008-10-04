// RUN: clang -fsyntax-only -verify %s
// XFAIL

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

/* expected-warning {{missing '(' after '#pragma pack'}}*/ #pragma pack 10
#pragma pack()
#pragma pack(8)

/*expected-warning {{unknown action for '#pragma pack'}}*/ #pragma pack(hello) 
#pragma pack(push)
#pragma pack(pop)

/* expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}*/ #pragma pack(push,)
/* expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}*/ #pragma pack(push,)
/* expected-warning {{malformed '#pragma pack', expected '#pragma pack(pop}}*/  #pragma pack(pop,) 

#pragma pack(push,i)
/* expected-warning {{malformed '#pragma pack', expected}}*/ #pragma pack(push,i, 
/* expected-warning {{malformed '#pragma pack', expected}}*/ #pragma pack(push,i,) 

#pragma pack(push,8)
/* expected-warning {{malformed '#pragma pack', expected}}*/ #pragma pack(push,8, 
/* expected-warning {{malformed '#pragma pack', expected}}*/ #pragma pack(push,8,help) 
/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push,8,) 
/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push,i,8 
#pragma pack(push,i,8)

/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push 

_Pragma("pack(push)")
/* expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}*/ _Pragma("pack(push,)") 

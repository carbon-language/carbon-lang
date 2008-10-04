// RUN: clang -fsyntax-only -verify %s
// XFAIL

#pragma pack 10 // expected-warning {{missing '(' after '#pragma pack'}}
#pragma pack()
#pragma pack(8)

#pragma pack(hello) // expected-warning {{unknown action for '#pragma pack'}}
#pragma pack(push)
#pragma pack(pop)

#pragma pack(push,) // expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}
#pragma pack(push,) // expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}
#pragma pack(pop,) // expected-warning {{malformed '#pragma pack', expected '#pragma pack(pop}}

#pragma pack(push,i)
#pragma pack(push,i, // expected-warning {{malformed '#pragma pack', expected}}
#pragma pack(push,i,) // expected-warning {{malformed '#pragma pack', expected}}

#pragma pack(push,8)
#pragma pack(push,8, // expected-warning {{malformed '#pragma pack', expected}}
#pragma pack(push,8,help) // expected-warning {{malformed '#pragma pack', expected}}
#pragma pack(push,8,) // expected-warning {{missing ')' after '#pragma pack'}}
#pragma pack(push,i,8 // expected-warning {{missing ')' after '#pragma pack'}}
#pragma pack(push,i,8)

#pragma pack(push // expected-warning {{missing ')' after '#pragma pack'}}

_Pragma("pack(push)")
_Pragma("pack(push,)") // expected-warning {{malformed '#pragma pack', expected '#pragma pack(push}}

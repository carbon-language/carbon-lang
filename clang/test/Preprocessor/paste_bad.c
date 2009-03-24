// GCC PR 20077
// RUN: not clang-cc -E %s &&
// RUN: not clang-cc -E %s 2>&1 | grep error: | wc -l | grep 10

#define a   a ## ## /* { dg-error "end of a macro expansion" } */
#define b() b ## ## /* { dg-error "end of a macro expansion" } */
#define c   c ##    /* { dg-error "end of a macro expansion" } */
#define d() d ##    /* { dg-error "end of a macro expansion" } */


#define e   ## ## e /* { dg-error "end of a macro expansion" } */
#define f() ## ## f /* { dg-error "end of a macro expansion" } */
#define g   ## g    /* { dg-error "end of a macro expansion" } */
#define h() ## h    /* { dg-error "end of a macro expansion" } */
#define i   ##      /* { dg-error "end of a macro expansion" } */
#define j() ##      /* { dg-error "end of a macro expansion" } */


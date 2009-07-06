// RUN: clang -fsyntax-only -Xclang -verify %s
struct {unsigned x : 2;} x;
__typeof__((x.x+=1)+1) y;
__typeof__(x.x<<1) y;
int y;


struct { int x : 8; } x1;
long long y1;
__typeof__(((long long)x1.x + 1)) y1;
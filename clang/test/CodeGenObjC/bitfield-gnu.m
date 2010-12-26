// RUN: %clang -S -emit-llvm -fgnu-runtime -o %t %s
typedef enum { A1, A2 } A;
typedef struct { A a : 1; } B;
@interface Obj { B *b; } @end
@implementation Obj @end

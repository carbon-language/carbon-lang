// RUN: clang %s -emit-llvm -o %t
typedef struct { unsigned int i: 1; } c;
const c d = { 1 };

// PR2310
struct Token {
  unsigned n : 31;
};
void sqlite3CodeSubselect(){
  struct Token one = { 1 };
}


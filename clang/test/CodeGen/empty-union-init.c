// RUN: clang-cc -emit-llvm < %s -o -
// PR2419

struct Mem {
        union {
        } u;
};

struct Mem *columnMem(){
        static const struct Mem nullMem = { {} };
}



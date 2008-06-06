// RUN: clang -emit-llvm < %s -o -

struct Mem {
        union {
        } u;
};

struct Mem *columnMem(){
        static const struct Mem nullMem = { {} };
}



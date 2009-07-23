// RUN: clang-cc -emit-llvm -o - 

// PR4610
#pragma pack(4)
struct ref {
        struct ref *next;
} refs;

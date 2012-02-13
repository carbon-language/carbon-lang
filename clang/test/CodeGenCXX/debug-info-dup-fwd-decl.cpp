// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
// XFAIL: *
class Test
{
public:
    Test () : reserved (new data()) {}

    unsigned
    getID() const
    {
        return reserved->objectID;
    }
protected:
    struct data {
        unsigned objectID;
    };
    data* reserved;
};

Test t;

// CHECK: metadata !"data", metadata !7, i32 13, i64 32, i64 32, i32 0, i32 0
// CHECK: metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !5} ; [ DW_TAG_pointer_type ]
// CHECK-NOT: metadata !"data", metadata !7, i32 13, i64 0, i64 0, i32 0, i32 4,

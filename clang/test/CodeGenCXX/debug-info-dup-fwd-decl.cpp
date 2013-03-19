// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -fno-limit-debug-info %s -o - | FileCheck %s

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

// CHECK: ; [ DW_TAG_pointer_type ]
// CHECK: ; [ DW_TAG_structure_type ] [data]
// CHECK-NOT: ; [ DW_TAG_structure_type ] [data]

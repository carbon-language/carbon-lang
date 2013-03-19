// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

class MyClass
{
public:
    int add2(int j)
    {
        return add<2>(j);
    }
private:
    template <int i> int add(int j)
    {
        return i + j;
    }
};

MyClass m;

// CHECK: metadata [[C_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_class_type ] [MyClass]
// CHECK: [[C_MEM]] = metadata !{metadata {{.*}}, metadata [[C_TEMP:![0-9]*]], metadata {{.*}}}
// CHECK: [[C_TEMP]] = {{.*}} ; [ DW_TAG_subprogram ] [line 11] [private] [add<2>]

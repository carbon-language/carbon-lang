// RUN: %clang_cc1 -ast-print %s | FileCheck %s

// CHECK: r;
// CHECK-NEXT: (r->method());
struct MyClass
{
    void method() {}
};

struct Reference
{
    MyClass* object;
    MyClass* operator ->() { return object; }
};

int main()
{
    Reference r;
    (r->method());
}


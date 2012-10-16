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

// CHECK: if (int a = 1)
// CHECK:  while (int a = 1)
// CHECK:  switch (int a = 1)

void f()
{
    if (int a = 1) { }
    while (int a = 1) { }
    switch (int a = 1) { }
}


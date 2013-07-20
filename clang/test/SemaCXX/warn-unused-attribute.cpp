// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s
struct __attribute__((warn_unused)) Test
{
    Test();
    ~Test();
    void use();
};

struct TestNormal
{
    TestNormal();
};

int main()
{
   Test unused;         // expected-warning {{unused variable 'unused'}}
   Test used;
   TestNormal normal;
   used.use();
}

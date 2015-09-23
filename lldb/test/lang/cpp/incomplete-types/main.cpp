
#include "length.h"

class Foo {
public:
    Foo(std::string x) : s(x) {}

private:
    std::string s;
};

class MyString : public std::string {
public:
    MyString(std::string x) : std::string(x) {}
};

int main()
{
    Foo f("qwerty");
    MyString s("qwerty");

    return length(s); // break here
}

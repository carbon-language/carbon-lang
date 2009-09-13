#include <iostream>

extern "C" void test();
extern std::string test2();

int main() {
    std::cout << "h";
    test();
    std::cout << test2() << '\n';
}

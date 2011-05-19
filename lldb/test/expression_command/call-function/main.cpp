#include <iostream>
#include <string>

int main (int argc, char const *argv[])
{
    std::string str = "Hello world";
    std::cout << str << std::endl;
    std::cout << str.c_str() << std::endl;
    // Please test these expressions while stopped at this line:
#if 0
    print str
    print str.c_str()
#endif
    return 0;
}

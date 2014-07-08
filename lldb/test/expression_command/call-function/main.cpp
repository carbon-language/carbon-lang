#include <iostream>
#include <string>

struct Five
{
    int number;
    const char *name;
};

Five
returnsFive()
{
    Five my_five = { 5, "five" };
    return my_five;
}

int main (int argc, char const *argv[])
{
    std::string str = "Hello world";
    std::cout << str << std::endl;
    std::cout << str.c_str() << std::endl;
    Five main_five = returnsFive();
#if 0
    print str
    print str.c_str()
#endif
    return 0; // Please test these expressions while stopped at this line:
}

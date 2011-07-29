#include <list>
#include <map>
#include <string>
#include <vector>
typedef std::vector<int> int_vect;
typedef std::vector<std::string> string_vect;

typedef std::list<int> int_list;
typedef std::list<std::string> string_list;

typedef std::map<int, int> intint_map;
typedef std::map<std::string, int> strint_map;
typedef std::map<int, std::string> intstr_map;
typedef std::map<std::string, std::string> strstr_map;

struct foo
{
    int a;
    int b;
    int c;
    int d;
    int e;
    int f;
    int g;
    int h;
    int i;
    int j;
    int k;
    int l;
    int m;
    int n;
    int o;
    int p;
    int q;
    int r;
    
    foo(int X) :
    a(X),
    b(X+1),
    c(X+3),
    d(X+5),
    e(X+7),
    f(X+9),
    g(X+11),
    h(X+13),
    i(X+15),
    j(X+17),
    k(X+19),
    l(X+21),
    m(X+23),
    n(X+25),
    o(X+27),
    p(X+29),
    q(X+31),
    r(X+33) {}
};

struct wrapint
{
    int x;
    wrapint(int X) : x(X) {}
};

int main()
{
    foo f00_1(0);
    foo *f00_ptr = new foo(12);
    
    f00_1.a++; // Set break point at this line.
    
    wrapint test_cast('A' +
               256*'B' +
               256*256*'C'+
               256*256*256*'D');
    
    int_vect numbers;
    numbers.push_back(1);
    numbers.push_back(12);
    numbers.push_back(123);
    numbers.push_back(1234);
    numbers.push_back(12345);
    numbers.push_back(123456);
    numbers.push_back(1234567);
    
    numbers.clear();
    
    numbers.push_back(7);

    string_vect strings;
    strings.push_back(std::string("goofy"));
    strings.push_back(std::string("is"));
    strings.push_back(std::string("smart"));
    
    strings.push_back(std::string("!!!"));
    
    strings.clear();
    
    int_list numbers_list;
    
    numbers_list.push_back(0x12345678);
    numbers_list.push_back(0x11223344);
    numbers_list.push_back(0xBEEFFEED);
    numbers_list.push_back(0x00ABBA00);
    numbers_list.push_back(0x0ABCDEF0);
    numbers_list.push_back(0x0CAB0CAB);
    
    numbers_list.clear();
    
    numbers_list.push_back(1);
    numbers_list.push_back(2);
    numbers_list.push_back(3);
    numbers_list.push_back(4);
    
    string_list text_list;
    text_list.push_back(std::string("goofy"));
    text_list.push_back(std::string("is"));
    text_list.push_back(std::string("smart"));
    
    text_list.push_back(std::string("!!!"));
    
    intint_map ii;
    
    ii[0] = 0;
    ii[1] = 1;
    ii[2] = 0;
    ii[3] = 1;
    ii[4] = 0;
    ii[5] = 1;
    ii[6] = 0;
    ii[7] = 1;
    ii[8] = 0;
    
    ii.clear();
    
    strint_map si;
    
    si["zero"] = 0;
    si["one"] = 1;
    si["two"] = 2;
    si["three"] = 3;
    si["four"] = 4;

    si.clear();
    
    intstr_map is;
    
    is[0] = "goofy";
    is[1] = "is";
    is[2] = "smart";
    is[3] = "!!!";
    
    is.clear();
    
    strstr_map ss;
    
    ss["ciao"] = "hello";
    ss["casa"] = "house";
    ss["gatto"] = "cat";
    ss["a Mac.."] = "..is always a Mac!";
    
    ss.clear();
    
    return 0;
}
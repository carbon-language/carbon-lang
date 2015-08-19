namespace N
{
    int n;
}

namespace
{
    int anon;
}

namespace Nested
{
    namespace
    {
        int nested;
    }
}

using namespace N;
using namespace Nested;

int main()
{
    n = 1;
    anon = 2;
    nested = 3;
    return 0; // break 0
}

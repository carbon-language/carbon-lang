#include <vector>

#include <stdio.h>
#include <stdint.h>

int main (int argc, char const *argv[], char const *envp[])
{
    std::vector<int> ints;
    std::vector<short> shorts;  
    for (int i=0; i<12; i++)
    {
        ints.push_back(i);
        shorts.push_back((short)i);
    }
    return 0;
}

// Linker initialized:
int getAB();
static int ab = getAB();
// Function local statics:
int countCalls();
static int one = countCalls();
// Constexpr:
int getCoolestInteger();
static int coolest_integer = getCoolestInteger();

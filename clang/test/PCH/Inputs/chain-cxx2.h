// Dependent header for C++ chained PCH test

// Overload function from primary
void f(int);

// Add function with different name
void f2();

// Reopen namespace
namespace ns {
  // Overload function from primary
  void g(int);

  // Add different name
  void g2();
}

// Specialize template from primary
template <>
struct S<int> { typedef int I; };

// RUN: %clang-cc1 %s -fexceptions

// Reduced from a crash on boost::interprocess's node_allocator_test.cpp.
namespace test0 {
  struct A { A(); ~A(); };
  struct V { V(const A &a = A()); ~V(); };

  template<int X> int vector_test()
  {
   A process_name;
   try {
     A segment;

     V *stdvector = new V();

     int x = 5, y = 7;
     if(x == y) return 1;
   }
   catch(int ex){
     return 1;
   }
   return 0;
}

int main ()
{
  return vector_test<0>();
}
}

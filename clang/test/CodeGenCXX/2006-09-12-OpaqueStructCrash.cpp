// RUN: %clang_cc1 -cxx-abi itanium -emit-llvm -o - %s
// RUN: %clang_cc1 -cxx-abi microsoft -fno-rtti -emit-llvm -o - %s

struct A {
   virtual ~A();
};

template <typename Ty>
struct B : public A {
   ~B () { delete [] val; }
private:
     Ty* val;
};

template <typename Ty>
struct C : public A {
   C ();
   ~C ();
};

template <typename Ty>
struct D : public A {
     D () {}
   private:
     B<C<Ty> > blocks;
};

template class D<double>;

// RUN: clang-cc -fsyntax-only -verify %s 
namespace A
{
    namespace B
    {
        struct base // expected-note{{object type}}
        {
            void x() {}
            void y() {}
        };
    }

    struct member
    {
        void foo();
    };

    struct middleman
    {
        member * operator->() { return 0; }
    };

    struct sub : B::base
    {
        void x() {}
        middleman operator->() { return middleman(); }
    };
}

struct bad
{
  int x();
};

namespace C
{
    void fun()
    {
        A::sub a;

        a.x();
    
        a.sub::x();
        a.base::x();

        a.B::base::x(); // expected-error{{use of undeclared identifier 'B'}}

        a.A::sub::x();
        a.A::B::base::x();

        a.bad::x(); // expected-error{{type 'struct bad' is not a direct or virtual base of ''struct A::sub''}}

        a->foo();
        a->member::foo();
        a->A::member::foo();
    }

    void fun2()
    {
        A::sub *a;

        a->x();

        a->sub::x();
        a->base::x();

        a->B::base::x(); // expected-error{{use of undeclared identifier 'B'}}

        a->A::sub::x();
        a->A::B::base::x();

        a->bad::x(); // expected-error{{type 'struct bad' is not a direct or virtual base of ''struct A::sub''}}

        (*a)->foo();
        (*a)->member::foo();
        (*a)->A::member::foo();
    }

    void fun3()
    {
        int i;
        i.foo(); // expected-error{{member reference base type 'int' is not a structure or union}}
    }

    void fun4a() {
      A::sub *a;
      
      typedef A::member base; // expected-note{{current scope}}
      a->base::x(); // expected-error{{ambiguous}}      
    }

    void fun4b() {
      A::sub *a;
      
      typedef A::B::base base;
      a->base::x();
    }
  
    template<typename T>
    void fun5()
    {
        T a;
        a.x();
        a->foo();

#if 0
        // FIXME: We need the notion of identifiers as dependent 
        // nested-name-specifiers without a prefix for this code to work.
      
        // Things that work for the wrong reason
        a.A::sub::x();
        a.A::B::base::x();
        a->A::member::foo();

        // Things that work, but shouldn't
        a.bad::x();

        // Things that fail, but shouldn't
        a.sub::x(); // xpected-error{{use of undeclared identifier 'sub'}}
        a.base::x(); // xpected-error{{use of undeclared identifier 'base'}}
        a.B::base::x(); // xpected-error{{use of undeclared identifier 'B'}}
        a->member::foo(); // xpected-error{{use of undeclared identifier 'member'}}
#endif
    }
}

// PR4703
struct a {
  int a;
  static int sa;
};

a a;

int a::sa = a.a;

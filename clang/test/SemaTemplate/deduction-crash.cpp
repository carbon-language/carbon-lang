// RUN: %clang_cc1 -fsyntax-only %s 2>&1| FileCheck %s

// PR7511

// Note that the error count below doesn't matter. We just want to
// make sure that the parser doesn't crash.
// CHECK: 15 errors
template<a>
struct int_;

template<a>
template<int,typename T1,typename>
struct ac
{
  typedef T1 ae
};

template<class>struct aaa
{
  typedef ac<1,int,int>::ae ae
};

template<class>
struct state_machine
{
  typedef aaa<int>::ae aaa;
  int start()
  {
    ant(0);
  }
  
  template<class>
  struct region_processing_helper
  {
    template<class,int=0>
    struct In;
    
    template<int my>
    struct In<a::int_<aaa::a>,my>;
        
    template<class Event>
    int process(Event)
    {
      In<a::int_<0> > a;
    }
  }
  template<class Event>
  int ant(Event)
  {
    region_processing_helper<int>* helper;
    helper->process(0)
  }
};

int a()
{
  state_machine<int> p;
  p.ant(0);
}

// RUN: %clang_cc1 -fsyntax-only %s 2>&1| FileCheck %s

// Note that the error count below doesn't matter. We just want to
// make sure that the parser doesn't crash.
// CHECK: 13 errors

// PR7511
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

// PR9974
template <int> struct enable_if;
template <class > struct remove_reference ;
template <class _Tp> struct remove_reference<_Tp&> ;

template <class > struct __tuple_like;

template <class _Tp, class _Up, int = __tuple_like<typename remove_reference<_Tp>::type>::value> 
struct __tuple_convertible;

struct pair
{
template<class _Tuple, int = enable_if<__tuple_convertible<_Tuple, pair>::value>::type> 
pair(_Tuple&& );
};

template <class> struct basic_ostream;

template <int> 
void endl( ) ;

extern basic_ostream<char> cout;

int operator<<( basic_ostream<char> , pair ) ;

void register_object_imp ( )
{
cout << endl<1>;
}

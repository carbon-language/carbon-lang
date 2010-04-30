// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5311
template<typename T>
class StringSwitch {
public:
  template<unsigned N>
  void Case(const char (&S)[N], const int & Value) {
  }
};

void test_stringswitch(int argc, char *argv[]) {
  (void)StringSwitch<int>();
}

namespace PR6986 {
  template<class Class,typename Type,Type Class::*> 
  struct non_const_member_base
  {
  };

  template<class Class,typename Type,Type Class::*PtrToMember> 
  struct member: non_const_member_base<Class,Type,PtrToMember>
  {
  };

  struct test_class
  {
    int int_member;
  };
  typedef member< test_class,const int,&test_class::int_member > ckey_m;
  void test()
  {
    ckey_m m;
  }
}

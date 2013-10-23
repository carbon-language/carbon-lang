typedef int using_decl_type;
int using_decl_var;

namespace UsingDecl {
  using ::using_decl_type;
  using ::using_decl_var;

  namespace A { typedef int inner; }
  using A::inner;
}

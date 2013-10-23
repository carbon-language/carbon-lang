namespace UsingDecl {
  namespace B { typedef int inner; }
  using B::inner;
}

#include "using-decl-a.h"

namespace UsingDecl {
  using ::using_decl_type;
  using ::using_decl_var;
}

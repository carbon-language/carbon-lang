// Define a few different kinds of types - no modules problems.

typedef int TypeInt;

typedef TypeInt NestedTypeInt;

struct TypeStruct {
  int Member;
};

class TypeClass {
public:
  TypeClass() : Member(0) {}
private:
  int Member;
};

/*
This currently doesn't work.  Can't handle same name in different namespaces.
namespace Namespace1 {
  class NamespaceClass {
  public:
    NamespaceClass() : Member(0) {}
  private:
    int Member;
  };
}

namespace Namespace2 {
  class NamespaceClass {
  public:
    NamespaceClass() : Member(0) {}
  private:
    int Member;
  };
}
*/


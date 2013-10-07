@import templates_top;

template<typename T> class Vector;

template<typename T> class Vector;

template<typename T> class List;
template<> class List<bool> {
public:
  void push_back(int);
};
namespace N {
  template<typename T> class Set;
}
namespace N {
  template<typename T> class Set {
  public:
    void insert(T);
  };
}

constexpr unsigned List<int>::*size_left = &List<int>::size;
List<int> list_left = { 0, 8 };
typedef List<int> ListInt_left;

template <typename T>
void pendingInstantiationEmit(T) {}
void triggerPendingInstantiation() {
  pendingInstantiationEmit(12);
  pendingInstantiationEmit(42.);
}

void redeclDefinitionEmit(){}

typedef Outer<int>::Inner OuterIntInner_left;

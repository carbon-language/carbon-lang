@import templates_top;

template<typename T> class Vector { 
public:
  void push_back(const T&);
};

template<typename T> class List;
template<> class List<bool> {
public:
  void push_back(int);
};

namespace N {
  template<typename T> class Set {
  public:
    void insert(T);
  };
}

constexpr unsigned List<int>::*size_right = &List<int>::size;
List<int> list_right = { 0, 12 };
typedef List<int> ListInt_right;

template <typename T>
void pendingInstantiationEmit(T) {}
void triggerPendingInstantiationToo() {
  pendingInstantiationEmit(12);
}

void redeclDefinitionEmit(){}

typedef Outer<int>::Inner OuterIntInner_right;

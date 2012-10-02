@__experimental_modules_import templates_top;

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

template <typename T>
void pendingInstantiation(T) {}
void triggerPendingInstantiation() {
  pendingInstantiation(12);
  pendingInstantiation(42.);
}

template<typename T> T f();
template<typename T> T f(T t) { return t; }
namespace N {
  template<typename T> T f();
  template<typename T> T f(T t) { return t; }
}

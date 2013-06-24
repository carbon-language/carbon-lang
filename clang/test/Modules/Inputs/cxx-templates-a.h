template<typename T> T f() { return T(); }
template<typename T> T f(T);
namespace N {
  template<typename T> T f() { return T(); }
  template<typename T> T f(T);
}

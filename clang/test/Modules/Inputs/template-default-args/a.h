BEGIN
template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T> struct C;
template<typename T> struct D;
template<typename T> struct E;
template<typename T = int> struct G;
template<typename T = int> struct H;
template<typename T> struct J {};
template<typename T = int> struct J;
struct K : J<> {};
template<typename T = void> struct L;
struct FriendL {
  template<typename T> friend struct L;
};
END

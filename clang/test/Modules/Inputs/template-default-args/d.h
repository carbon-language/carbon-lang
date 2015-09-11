BEGIN
template<typename T = void> struct L;
struct FriendL {
  template<typename T> friend struct L;
};
END

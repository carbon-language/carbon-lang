class HasFriends {
  friend void friend_1(HasFriends);
  friend void friend_2(HasFriends);
  void private_thing();
};

struct HasNontrivialDefaultConstructor {
  HasNontrivialDefaultConstructor() = default;
  HasNontrivialDefaultConstructor(int n = 0);

  // Ensure this class is not POD but is still trivially-copyable.
  // This is necessary to exercise the second static_assert below,
  // because GCC's spec for __has_trivial_constructor is absurd.
  int m;
private:
  int n;
};

static_assert(!__is_trivial(HasNontrivialDefaultConstructor), "");
static_assert(!__has_trivial_constructor(HasNontrivialDefaultConstructor), "");

void *operator new[](__SIZE_TYPE__);

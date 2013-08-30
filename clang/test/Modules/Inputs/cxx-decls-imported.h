class HasFriends {
  friend void friend_1(HasFriends);
  friend void friend_2(HasFriends);
  void private_thing();
};

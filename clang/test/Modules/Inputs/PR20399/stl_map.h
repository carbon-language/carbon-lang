namespace std
{
  template<typename _Iterator>
  class reverse_iterator {};

  template<typename _Iterator>
  inline int*
  operator-(const int& __x, const reverse_iterator<_Iterator>& __y) {};

  template<typename _Tp>
  struct _Rb_tree_iterator
  {
    typedef _Rb_tree_iterator<_Tp>        _Self;
  };

  template <typename _Key, typename _Tp >
  class map
  {
  public:
    typedef _Rb_tree_iterator<int>        iterator;

    template<typename _K1, typename _T1>
    friend bool operator<(const map<_K1, _T1>&, const map<_K1, _T1>&);
  };
} // namespace std

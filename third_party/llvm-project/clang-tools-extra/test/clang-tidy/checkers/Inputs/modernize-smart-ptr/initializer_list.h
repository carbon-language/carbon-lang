namespace std {
typedef decltype(sizeof(int)) size_t;

template <class _E> class initializer_list {
  const _E *__begin_;
  size_t __size_;

  initializer_list(const _E *__b, size_t __s) : __begin_(__b), __size_(__s) {}

public:
  typedef _E value_type;
  typedef const _E &reference;
  typedef const _E &const_reference;
  typedef size_t size_type;

  typedef const _E *iterator;
  typedef const _E *const_iterator;

  initializer_list() : __begin_(nullptr), __size_(0) {}

  size_t size() const { return __size_; }
  const _E *begin() const { return __begin_; }
  const _E *end() const { return __begin_ + __size_; }
};

template <class _E>
class vector {
 public:
  vector(initializer_list<_E> init);
  ~vector();
};
} // namespace std

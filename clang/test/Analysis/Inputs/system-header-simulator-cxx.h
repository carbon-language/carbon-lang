// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

namespace std {
  template <class T1, class T2>
  struct pair {
    T1 first;
    T2 second;
    
    pair() : first(), second() {}
    pair(const T1 &a, const T2 &b) : first(a), second(b) {}
    
    template<class U1, class U2>
    pair(const pair<U1, U2> &other) : first(other.first), second(other.second) {}
  };
  
  typedef __typeof__(sizeof(int)) size_t;
  
  template<typename T>
  class vector {
    T *_start;
    T *_finish;
    T *_end_of_storage;
  public:
    vector() : _start(0), _finish(0), _end_of_storage(0) {}
    ~vector();
    
    size_t size() const {
      return size_t(_finish - _start);
    }
    
    void push_back();
    T pop_back();

    T &operator[](size_t n) {
      return _start[n];
    }
    
    const T &operator[](size_t n) const {
      return _start[n];
    }
    
    T *begin() { return _start; }
    const T *begin() const { return _start; }

    T *end() { return _finish; }
    const T *end() const { return _finish; }
  };
  
  class exception {
  public:
    exception() throw();
    virtual ~exception() throw();
    virtual const char *what() const throw() {
      return 0;
    }
  };

  class bad_alloc : public exception {
    public:
    bad_alloc() throw();
    bad_alloc(const bad_alloc&) throw();
    bad_alloc& operator=(const bad_alloc&) throw();
    virtual const char* what() const throw() {
      return 0;
    }
  };

  struct nothrow_t {};

  extern const nothrow_t nothrow;

  template<class InputIter, class OutputIter>
  OutputIter copy(InputIter II, InputIter IE, OutputIter OI) {
    while (II != IE)
      *OI++ = *II++;
    return OI;
  }

  struct input_iterator_tag { };
  struct output_iterator_tag { };
  struct forward_iterator_tag : public input_iterator_tag { };
  struct bidirectional_iterator_tag : public forward_iterator_tag { };
  struct random_access_iterator_tag : public bidirectional_iterator_tag { };

  template <class _Tp>
  class allocator {};

  template <class _Tp, class _Alloc>
  class __list_imp
  {};

  template <class _Tp, class _Alloc = allocator<_Tp> >
  class list
  : private __list_imp<_Tp, _Alloc>
  {
  public:
    void pop_front() {
      // Fake use-after-free.
      // No warning is expected as we are suppressing warning comming
      // out of std::list.
      int z = 0;
      z = 5/z;
    }
    bool empty() const;
  };

}

void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new[](std::size_t, const std::nothrow_t&) throw();
void operator delete(void*, const std::nothrow_t&) throw();
void operator delete[](void*, const std::nothrow_t&) throw();

void* operator new (std::size_t size, void* ptr) throw() { return ptr; };
void* operator new[] (std::size_t size, void* ptr) throw() { return ptr; };
void operator delete (void* ptr, void*) throw() {};
void operator delete[] (void* ptr, void*) throw() {};

// RUN: clang-cc -fsyntax-only -verify %s

struct nil { };

template<typename Head, typename Tail = nil>
struct cons { 
  typedef Head head;
  typedef Tail tail;
};

// metaprogram that computes the length of a list
template<typename T> struct length;

template<typename Head, typename Tail>
struct length<cons<Head, Tail> > {
  static const unsigned value = length<Tail>::value + 1;
};

template<>
struct length<nil> {
  static const unsigned value = 0;
};

typedef cons<unsigned char, 
             cons<unsigned short, 
                  cons<unsigned int,
                       cons<unsigned long> > > > unsigned_inttypes;
int length0[length<unsigned_inttypes>::value == 4? 1 : -1];

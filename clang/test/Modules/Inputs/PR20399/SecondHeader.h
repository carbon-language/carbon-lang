#ifndef SECONDHEADER
#define SECONDHEADER

#include "vector"

class Collection {
  template <class T> struct Address { };
};

template <> struct Collection::Address<std::vector<bool> >
   : public Collection::Address<std::vector<bool>::iterator> { };

#endif

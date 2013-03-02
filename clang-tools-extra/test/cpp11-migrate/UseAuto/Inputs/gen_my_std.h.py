#!/usr/bin/python

# Each std container is represented below. To test the various ways in which
# a type may be defined, the containers are split into categories:
# * Define iterator types with typedefs
# * Define iterator types as nested classes
# * Define iterator types with using declarations
#
# Further, one class in each category is chosen to be defined in a way mimicing
# libc++: The container is actually defined in a different namespace (std::_1
# is used here) and then imported into the std namespace with a using
# declaration. This is controlled with the 'using' key in the dictionary
# describing each container.
typedef_containers = [
  {"name" : "array",
   "using" : True},
  {"name" : "deque",
   "using" : False},
  {"name" : "forward_list",
   "using" : False},
  {"name" : "list",
   "using" : False},
  {"name" : "vector",
   "using" : False}
]
subclass_containers = [
  {"name" : "map",
   "using" : True},
  {"name" : "multimap",
   "using" : False},
  {"name" : "set",
   "using" : False},
  {"name" : "multiset",
   "using" : False},
]
using_containers = [
  {"name" : "unordered_map",
   "using" : True},
  {"name" : "unordered_multimap",
   "using" : False},
  {"name" : "unordered_set",
   "using" : False},
  {"name" : "unordered_multiset",
   "using" : False},
  {"name" : "queue",
   "using" : False},
  {"name" : "priority_queue",
   "using" : False},
  {"name" : "stack",
   "using" : False}
]


# Every class requires these functions.
iterator_generators = """
  iterator begin() { return iterator(); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(); }
  const_iterator end() const { return const_iterator(); }

  reverse_iterator rbegin() { return reverse_iterator(); }
  reverse_iterator rend() { return reverse_iterator(); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(); }
  const_reverse_iterator rend() const { return const_reverse_iterator(); }
"""


# Convenience function for nested class definition within a special namespace
# to mimic libc++ style std container definitions.
def outputClassDef(Definition, ClassName, Import):
  if Import:
    print "namespace _1 {"

  print Definition

  if Import:
    print """
} // namespace _1
using _1::%s;""" % ClassName


# Output preamble and common functionality
print """
//===-----------------------------------------------------------*- C++ -*--===//
//
// This file was automatically generated from gen_my_std.h.py by the build
// system as a dependency for cpp11-migrate's test suite.
//
// This file contains a shell implementation of std containers and iterators for
// testing the use-auto transform of cpp11-migrate. All std containers and
// iterators are present. Container and iterator implementations vary to cover
// various ways the std container and iterator types are made available:
//
// Variations for how iterator types are presented:
// * Typedef (array, deque, forward_list, list, vector)
// * Nested class (map, multimap, set, multiset)
// * Using declaration {unordered_} X {map, multimap, set, multiset}
//
// Variations for how container types are presented:
// * Defined directly in namespace std
// * Imported into namespace std with using declarations (a la libc++).
//
//===----------------------------------------------------------------------===//

namespace internal {

template <typename T, int i>
struct iterator_wrapper {
  iterator_wrapper() {}

  // These are required for tests using iteration statements.
  bool operator!=(const iterator_wrapper<T, i>&) { return false; }
  iterator_wrapper& operator++() { return *this; }
  typename T::value_type operator*() { return typename T::value_type(); }
};

template <typename T>
class iterator_provider {
public:
  class iterator {
  public:
    iterator() {}
    iterator(const iterator&) {}
  };
  class const_iterator {
  public:
    const_iterator(int i=0) {}
    const_iterator(const iterator &) {}
    const_iterator(const const_iterator &) {}
    operator iterator() { return iterator(); }
  };
  class reverse_iterator {};
  class const_reverse_iterator {};
};

} // namespace internal

namespace std {""".lstrip() # Take off leading newline

for c in typedef_containers:
  Definition = """
template <typename T>
class %(0)s {
public:
  typedef T value_type;
  typedef typename internal::iterator_wrapper<%(0)s<T>, 0> iterator;
  typedef typename internal::iterator_wrapper<%(0)s<T>, 1> const_iterator;
  typedef typename internal::iterator_wrapper<%(0)s<T>, 3> reverse_iterator;
  typedef typename internal::iterator_wrapper<%(0)s<T>, 2> const_reverse_iterator;

  %(0)s() {}
  %(1)s};""" % {'0': c['name'], '1': iterator_generators}

  outputClassDef(Definition, c['name'], c['using'])

for c in subclass_containers:
  Definition = """
template <typename T>
class %(0)s {
public:
  class iterator {};
  class const_iterator {};
  class reverse_iterator {};
  class const_reverse_iterator {};

  %(0)s() {}
  %(1)s};""" % {'0': c['name'], '1': iterator_generators}

  outputClassDef(Definition, c['name'], c['using'])

for c in using_containers:
  Definition = """
template <typename T>
class %(0)s : internal::iterator_provider<%(0)s<T> > {
public:
  using typename internal::iterator_provider<%(0)s<T> >::iterator;
  using typename internal::iterator_provider<%(0)s<T> >::const_iterator;
  using typename internal::iterator_provider<%(0)s<T> >::reverse_iterator;
  using typename internal::iterator_provider<%(0)s<T> >::const_reverse_iterator;

  %(0)s() {}
  %(1)s};""" % {'0': c['name'], '1': iterator_generators}

  outputClassDef(Definition, c['name'], c['using'])

print "} // namespace std"

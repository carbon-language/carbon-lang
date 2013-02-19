#!/usr/bin/python

typedef_containers = [
    "array",
    "deque",
    "forward_list",
    "list",
    "vector"
]
subclass_containers = [
    "map",
    "multimap",
    "set",
    "multiset",
]
using_containers = [
    "unordered_map",
    "unordered_multimap",
    "unordered_set",
    "unordered_multiset",
    "queue",
    "priority_queue",
    "stack"
]

print """namespace internal {

template <typename T, int i>
struct iterator_wrapper {
};

template <typename T>
class iterator_provider {
public:
  class iterator {};
  class const_iterator {};
  class reverse_iterator {};
  class const_reverse_iterator {};
};

} // namespace internal

namespace std {"""

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

for c in typedef_containers:
  print """
template <typename T>
class {0} {{
public:
  typedef typename internal::iterator_wrapper<{0}<T>, 0> iterator;
  typedef typename internal::iterator_wrapper<{0}<T>, 1> const_iterator;
  typedef typename internal::iterator_wrapper<{0}<T>, 3> reverse_iterator;
  typedef typename internal::iterator_wrapper<{0}<T>, 2> const_reverse_iterator;

  {0}() {{}}
  {1}}};""".format(c, iterator_generators)

for c in subclass_containers:
  print """
template <typename T>
class {0} {{
public:
  class iterator {{}};
  class const_iterator {{}};
  class reverse_iterator {{}};
  class const_reverse_iterator {{}};

  {0}() {{}}
  {1}}};""".format(c, iterator_generators)

for c in using_containers:
  print """
template <typename T>
class {0} : internal::iterator_provider<{0}<T> > {{
public:
  using typename internal::iterator_provider<{0}<T> >::iterator;
  using typename internal::iterator_provider<{0}<T> >::const_iterator;
  using typename internal::iterator_provider<{0}<T> >::reverse_iterator;
  using typename internal::iterator_provider<{0}<T> >::const_reverse_iterator;

  {0}() {{}}
  {1}}};""".format(c, iterator_generators)

print "} // namespace std"

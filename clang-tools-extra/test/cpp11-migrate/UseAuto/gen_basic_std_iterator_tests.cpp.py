#!/usr/bin/python

containers = [
    "array",
    "deque",
    "forward_list",
    "list",
    "vector",
    "map",
    "multimap",
    "set",
    "multiset",
    "unordered_map",
    "unordered_multimap",
    "unordered_set",
    "unordered_multiset",
    "queue",
    "priority_queue",
    "stack"
]

print """
//===----------------------------------------------------------------------===//
//
// This file was automatically generated from
// gen_basic_std_iterator_tests.cpp.py by the build system as a dependency for
// cpp11-migrate's test suite.
//
// This file contains basic positive tests for the use-auto transform's ability
// to replace standard iterators. Variables considered:
// * All std container names
// * All std iterator names
//
//===----------------------------------------------------------------------===//

// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
#include "my_std.h"

int main(int argc, char **argv) {""".lstrip() # Strip leading newline

for c in containers:
  print """
  {
    std::%(0)s<int> C;
    std::%(0)s<int>::iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }
  {
    std::%(0)s<int> C;
    std::%(0)s<int>::reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }
  {
    const std::%(0)s<int> C;
    std::%(0)s<int>::const_iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }
  {
    const std::%(0)s<int> C;
    std::%(0)s<int>::const_reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }""" % {"0": c}

print """
  return 0;
}"""

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

print """// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- --std=c++11 -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *
#include "my_std.h"

int main(int argc, char **argv) {"""

for c in containers:
  print """
  {{
    std::{0}<int> C;
    std::{0}<int>::iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }}
  {{
    std::{0}<int> C;
    std::{0}<int>::reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }}
  {{
    const std::{0}<int> C;
    std::{0}<int>::const_iterator I = C.begin();
    // CHECK: auto I = C.begin();
  }}
  {{
    const std::{0}<int> C;
    std::{0}<int>::const_reverse_iterator I = C.rbegin();
    // CHECK: auto I = C.rbegin();
  }}""".format(c)

print """
  return 0;
}"""

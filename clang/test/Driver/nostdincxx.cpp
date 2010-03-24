// RUN: %clangxx -nostdinc++ %s 2>&1 | FileCheck %s

// CHECK: file not found
#include <vector> 

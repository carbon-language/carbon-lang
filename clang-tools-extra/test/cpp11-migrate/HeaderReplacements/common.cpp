// This is just a dummy run command to keep lit happy. Tests for this file are
// in main.cpp
// RUN: true

#include "common.h"

void func1(int &I) {
}

void func2() {
  container C1;
  container C2;
  for (container::iterator I = C1.begin(), E = C1.end(); I != E; ++I) {
    C2.push_back(*I);
  }
}

//===- llvm/unittest/Support/DynamicLibrary/PipSqueak.cxx -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PipSqueak.h"
#include <string>

struct Global {
  std::string *Str;
  Global() : Str(nullptr) {}
  ~Global() {
    if (Str)
      *Str = "Global::~Global";
  }
};

struct Local {
  std::string &Str;
  Local(std::string &S) : Str(S) { Str = "Local::Local"; }
  ~Local() { Str = "Local::~Local"; }
};

static Global Glb;

extern "C" PIPSQUEAK_EXPORT void SetStrings(std::string &GStr,
                                            std::string &LStr) {
  static Local Lcl(LStr);
  Glb.Str = &GStr;
}

extern "C" PIPSQUEAK_EXPORT const char *TestA() { return "LibCall"; }

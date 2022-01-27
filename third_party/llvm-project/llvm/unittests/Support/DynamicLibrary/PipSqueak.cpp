//===- llvm/unittest/Support/DynamicLibrary/PipSqueak.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PipSqueak.h"

struct Global {
  std::string *Str;
  std::vector<std::string> *Vec;
  Global() : Str(nullptr), Vec(nullptr) {}
  ~Global() {
    if (Str) {
      if (Vec)
        Vec->push_back(*Str);
      *Str = "Global::~Global";
    }
  }
};

static Global Glb;

struct Local {
  std::string &Str;
  Local(std::string &S) : Str(S) {
    Str = "Local::Local";
    if (Glb.Str && !Glb.Str->empty())
      Str += std::string("(") + *Glb.Str + std::string(")");
  }
  ~Local() { Str = "Local::~Local"; }
};


extern "C" PIPSQUEAK_EXPORT void SetStrings(std::string &GStr,
                                            std::string &LStr) {
  Glb.Str = &GStr;
  static Local Lcl(LStr);
}

extern "C" PIPSQUEAK_EXPORT void TestOrder(std::vector<std::string> &V) {
  Glb.Vec = &V;
}

#define PIPSQUEAK_TESTA_RETURN "LibCall"
#include "ExportedFuncs.cpp"

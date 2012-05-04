//===------ polly/CodeGeneration.h - The Polly code generator *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGENERATION_H
#define POLLY_CODEGENERATION_H

#include "polly/Config/config.h"
#ifdef CLOOG_FOUND

namespace polly {
  extern bool EnablePollyVector;
}

#endif // CLOOG_FOUND
#endif // POLLY_CODEGENERATION_H


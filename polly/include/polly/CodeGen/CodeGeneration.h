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

#include "isl/set.h"
#include "isl/map.h"

namespace polly {
enum VectorizerChoice {
  VECTORIZER_NONE,
  VECTORIZER_STRIPMINE,
  VECTORIZER_POLLY,
};
extern VectorizerChoice PollyVectorizerChoice;

enum CodeGenChoice { CODEGEN_ISL, CODEGEN_NONE };
extern CodeGenChoice PollyCodeGenChoice;

/// @brief Flag to turn on/off annotation of alias scopes.
extern bool PollyAnnotateAliasScopes;
}

#endif // POLLY_CODEGENERATION_H

//===- Codegen/IRBuilder.h - The IR builder used by Polly -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The Polly IRBuilder file contains Polly specific extensions for the
// IRBuilder.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGEN_IRBUILDER_H
#define POLLY_CODEGEN_IRBUILDER_H

#include "llvm/IR/IRBuilder.h"

namespace polly {
typedef llvm::IRBuilder<> PollyIRBuilder;
}
#endif

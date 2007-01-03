//===-- Streams.cpp - Wrappers for iostreams ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper for the std::cout and std::cerr I/O streams.
// It prevents the need to include <iostream> to each file just to get I/O.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Streams.h"
#include <iostream>
using namespace llvm;

OStream llvm::cout(std::cout);
OStream llvm::cerr(std::cerr);
IStream llvm::cin(std::cin);

//===--- Sema.cpp - Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis 
// out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "flang/Basic/Version.h"
#include "flang/Sema/Scope.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

using namespace flang;
using namespace sema;


//===- Scope.cpp - Lexical scope information --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Scope class, which is used for recording
// information about a lexical scope.
//
//===----------------------------------------------------------------------===//

#include "flang/Sema/Sema.h"

using namespace flang;
using namespace sema;

Sema::Sema()
{
}



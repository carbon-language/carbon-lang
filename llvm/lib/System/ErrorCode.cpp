//===- ErrorCode.cpp - Define the ErrorCode class ---------------*- C++ -*-===//
//
// Copyright (C) 2004 eXtensible Systems, Inc. All Rights Reserved.
//
// This program is open source software; you can redistribute it and/or modify
// it under the terms of the University of Illinois Open Source License. See
// LICENSE.TXT (distributed with this software) for details.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
//
//===----------------------------------------------------------------------===//
// 
// This file defines the members of the llvm::sys::ErrorCode class. This class
// is used to hold operating system and other error codes in a platform 
// agnostic manner.
//
//===----------------------------------------------------------------------===//
/// @file lib/System/ErrorCode.h
/// @author Reid Spencer <raspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Declares the llvm::sys::ErrorCode class.
//===----------------------------------------------------------------------===//

#include "llvm/System/ErrorCode.h"

namespace llvm {
namespace sys {

}
}

#include "linux/ErrorCode.cpp"

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

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
// This file defines the linux specific members of the llvm::sys::ErrorCode 
// class. 
//
//===----------------------------------------------------------------------===//
/// @file lib/System/ErrorCode.h
/// @author Reid Spencer <raspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Declares the linux specific methods of llvm::sys::ErrorCode class.
//===----------------------------------------------------------------------===//

namespace llvm {
namespace sys {

std::string 
ErrorCode::description() const throw()
{
  switch (domain()) {
    case OSDomain:
      char buffer[1024];
      if (0 != strerror_r(index(),buffer,1024) )
        return "<Error Message Unavalabile>";
      return buffer;

    case SystemDomain:
      switch (index()) {
        case ERR_SYS_INVALID_ARG: 
          return "Invalid argument to lib/System call";
        default:
          return "Unknown lib/System Error";
      }
      break;

    default:
      return "Unknown Error";
  }
}

}
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

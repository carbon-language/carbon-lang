//===- Path.cpp - Path Operating System Concept -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
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
// This file implements the common Path concept for a variety of platforms.
// A path is simply the name of some file system storage place. Paths can be
// either directories or files.
//
//===----------------------------------------------------------------------===//
/// @file lib/System/Path.cpp
/// @author Reid Spencer <raspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Defines the llvm::sys::Path class.
//===----------------------------------------------------------------------===//

#include "llvm/System/Path.h"

namespace llvm {
namespace sys {

ErrorCode
Path::append_directory( const std::string& dirname ) throw() {
  this->append( dirname );
  make_directory();
  return NOT_AN_ERROR;
}

ErrorCode
Path::append_file( const std::string& filename ) throw() {
  this->append( filename );
  return NOT_AN_ERROR;
}

ErrorCode
Path::create( bool create_parents)throw() {
  ErrorCode result ( NOT_AN_ERROR );
  if ( is_directory() ) {
    if ( create_parents ) {
      result = this->create_directories( );
    } else {
      result = this->create_directory( );
    }
  } else if ( is_file() ) {
    if ( create_parents ) {
      result = this->create_directories( );
    }
    if ( result ) {
      result = this->create_file( );
    }
  } else {
    result = ErrorCode(ERR_SYS_INVALID_ARG);
  }
  return result;
}

ErrorCode
Path::remove() throw() {
  ErrorCode result( NOT_AN_ERROR );
  if ( is_directory() ) {
    if ( exists() ) 
      this->remove_directory( );
  } else if ( is_file() ) {
    if ( exists() ) this->remove_file( );
  } else {
    result = ErrorCode(ERR_SYS_INVALID_ARG);
  }
  return result;
}

}
}

// Include the platform specific portions of this class
#include "linux/Path.cpp" 

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

//===-- Path.cpp - Implement OS Path Concept --------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Path concept.
//
//===----------------------------------------------------------------------===//
#include "llvm/System/Path.h"

namespace llvm {
namespace sys {

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

bool
Path::is_valid() const {
  if ( empty() ) return false;
  return true;
}

void 
Path::fill( char* buffer, unsigned bufflen ) const {
  unsigned pathlen = length();
  assert( bufflen > pathlen && "Insufficient buffer size" );
  unsigned copylen = pathlen <? (bufflen - 1);
  this->copy(buffer, copylen, 0 );
  buffer[ copylen ] = 0;
}

void
Path::make_directory() {
  char end[2];
  end[0] = '/';
  end[1] = 0;
  if ( empty() )
    this->assign( end );
  else if ( (*this)[length()-1] != '/')
    this->append( end );
}

void
Path::make_file() {
  if ( (*this)[length()-1] == '/')
    this->erase( this->length()-1, 1 );
}

// Include the truly platform-specific parts of this class.
#include "platform/Path.cpp"
}
}

// vim: sw=2

//===- llvm/System/Win32/Mutex.inc - Win32 Mutex Implementation -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) Mutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

namespace llvm
{
using namespace sys;

Mutex::Mutex( bool recursive)
{
}

Mutex::~Mutex()
{
}

bool 
Mutex::acquire()
{
}

bool 
Mutex::release()
{
}

bool 
Mutex::tryacquire( void )
{
}

}

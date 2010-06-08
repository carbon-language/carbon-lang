//===-- SBType.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBType_h_
#define LLDB_SBType_h_

#include <LLDB/SBDefines.h>

namespace lldb {

class SBType
{
public:

    static bool
    IsPointerType (void *opaque_type);

private:

};


} // namespace lldb

#endif // LLDB_SBType_h_

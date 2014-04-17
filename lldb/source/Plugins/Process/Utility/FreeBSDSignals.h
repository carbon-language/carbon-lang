//===-- FreeBSDSignals.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_FreeBSDSignals_H_
#define liblldb_FreeBSDSignals_H_

// Project includes
#include "lldb/Target/UnixSignals.h"

/// FreeBSD specific set of Unix signals.
class FreeBSDSignals
    : public lldb_private::UnixSignals
{
public:
    FreeBSDSignals();

private:
    void
    Reset();
};

#endif // liblldb_FreeBSDSignals_H_

//===-- SBFunction.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFunction_h_
#define LLDB_SBFunction_h_

#include <LLDB/SBDefines.h>

namespace lldb {

class SBFunction
{
public:

    SBFunction ();

    ~SBFunction ();

    bool
    IsValid () const;

    const char *
    GetName() const;

    const char *
    GetMangledName () const;

#ifndef SWIG
    bool
    operator == (const lldb::SBFunction &rhs) const;

    bool
    operator != (const lldb::SBFunction &rhs) const;
#endif

private:
    friend class SBFrame;
    friend class SBSymbolContext;

    SBFunction (lldb_private::Function *lldb_object_ptr);


    lldb_private::Function *m_lldb_object_ptr;
};


} // namespace lldb

#endif // LLDB_SBFunction_h_

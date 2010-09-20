//===-- SBSymbol.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbol_h_
#define LLDB_SBSymbol_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBSymbol
{
public:

    SBSymbol ();

    ~SBSymbol ();

    bool
    IsValid () const;


    const char *
    GetName() const;

    const char *
    GetMangledName () const;

#ifndef SWIG
    bool
    operator == (const lldb::SBSymbol &rhs) const;

    bool
    operator != (const lldb::SBSymbol &rhs) const;
#endif

    bool
    GetDescription (lldb::SBStream &description);

    // The following function gets called by Python when a user tries to print
    // an object of this class.  It takes no arguments and returns a
    // PyObject * representing a char * (and it must be named "__repr__");

    PyObject *
    __repr__ ();

private:
    friend class SBSymbolContext;

    SBSymbol (lldb_private::Symbol *lldb_object_ptr);

    lldb_private::Symbol *m_opaque_ptr;
};


} // namespace lldb

#endif // LLDB_SBSymbol_h_

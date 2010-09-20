//===-- SBCommandReturnObject.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandReturnObject_h_
#define LLDB_SBCommandReturnObject_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBCommandReturnObject
{
public:

    SBCommandReturnObject ();

    ~SBCommandReturnObject ();

    bool
    IsValid() const;

    const char *
    GetOutput ();

    const char *
    GetError ();

    size_t
    PutOutput (FILE *fh);

    size_t
    GetOutputSize ();

    size_t
    GetErrorSize ();

    size_t
    PutError (FILE *fh);

    void
    Clear();

    lldb::ReturnStatus
    GetStatus();

    bool
    Succeeded ();

    bool
    HasResult ();

    void
    AppendMessage (const char *message);

    bool
    GetDescription (lldb::SBStream &description);

    // The following function gets called by Python when a user tries to print
    // an object of this class.  It takes no arguments and returns a
    // PyObject * representing a char * (and it must be named "__repr__");

    PyObject *
    __repr__ ();

protected:
    friend class SBCommandInterpreter;
    friend class SBOptions;


#ifndef SWIG

    lldb_private::CommandReturnObject *
    operator->() const;

    lldb_private::CommandReturnObject *
    get() const;

    lldb_private::CommandReturnObject &
    operator*() const;

    lldb_private::CommandReturnObject &
    ref() const;

#endif
    void
    SetLLDBObjectPtr (lldb_private::CommandReturnObject *ptr);

 private:
    std::auto_ptr<lldb_private::CommandReturnObject> m_opaque_ap;
};

} // namespace lldb

#endif        // LLDB_SBCommandReturnObject_h_

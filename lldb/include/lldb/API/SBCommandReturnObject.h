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

#include <LLDB/SBDefines.h>

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

protected:
    friend class SBCommandInterpreter;
    friend class SBOptions;

    lldb_private::CommandReturnObject *
    GetLLDBObjectPtr();

    lldb_private::CommandReturnObject &
    GetLLDBObjectRef();

    void
    SetLLDBObjectPtr (lldb_private::CommandReturnObject *ptr);

 private:
    std::auto_ptr<lldb_private::CommandReturnObject> m_return_object_ap;
};

} // namespace lldb

#endif        // LLDB_SBCommandReturnObject_h_

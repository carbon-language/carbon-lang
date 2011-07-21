//===-- SWIG Interface for SBCommandReturnObject ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a container which holds the result from command execution.
It works with SBCommandInterpreter.HandleCommand() to encapsulate the result
of command execution.

See SBCommandInterpreter for example usage of SBCommandReturnObject."
) SBCommandReturnObject;
class SBCommandReturnObject
{
public:

    SBCommandReturnObject ();

    SBCommandReturnObject (const lldb::SBCommandReturnObject &rhs);

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
    
    void
    SetImmediateOutputFile (FILE *fh);
    
    void
    SetImmediateErrorFile (FILE *fh);
};

} // namespace lldb

//===-- SWIG Interface for SBThreadCollection -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

%feature("docstring",
"Represents a collection of SBThread objects."
) SBThreadCollection;
class SBThreadCollection
{
public:
    
    SBThreadCollection ();
    
    SBThreadCollection (const SBThreadCollection &rhs);
    
    ~SBThreadCollection ();
    
    bool
    IsValid () const;
    
    size_t
    GetSize ();
    
    lldb::SBThread
    GetThreadAtIndex (size_t idx);
    
};
    
} // namespace lldb

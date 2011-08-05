//===-- SWIG Interface for SBType -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBType
{
public:
    SBType (const SBType &rhs);

    ~SBType ();

    bool
    IsValid() const;

    size_t
    GetByteSize() const;

    bool
    IsPointerType() const;

    bool
    IsReferenceType() const;

    SBType
    GetPointerType() const;

    SBType
    GetPointeeType() const;

    SBType
    GetReferenceType() const;

    SBType
    GetDereferencedType() const;

    SBType
    GetBasicType(lldb::BasicType type) const;

    const char*
    GetName();
};

%feature("docstring",
"Represents a list of SBTypes.  The FindTypes() method of SBTarget/SBModule
returns a SBTypeList.

SBTypeList supports SBType iteration. For example,

main.cpp:

class Task {
public:
    int id;
    Task *next;
    Task(int i, Task *n):
        id(i),
        next(n)
    {}
};

...

find_type.py:

        # Get the type 'Task'.
        type_list = target.FindTypes('Task')
        self.assertTrue(len(type_list) == 1)
        # To illustrate the SBType iteration.
        for type in type_list:
            # do something with type

...
") SBTypeList;
class SBTypeList
{
public:
    SBTypeList();

    void
    Append(const SBType& type);

    SBType
    GetTypeAtIndex(int index);

    int
    GetSize();

    ~SBTypeList();
};

} // namespace lldb

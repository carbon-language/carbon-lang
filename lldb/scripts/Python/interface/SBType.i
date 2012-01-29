//===-- SWIG Interface for SBType -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
"Represents a member of a type in lldb.
") SBTypeMember;

class SBTypeMember
{
public:
    SBTypeMember ();

    SBTypeMember (const lldb::SBTypeMember& rhs);

    ~SBTypeMember();

    bool
    IsValid() const;

    const char *
    GetName ();

    lldb::SBType
    GetType ();

    uint64_t
    GetOffsetInBytes();
    
    uint64_t
    GetOffsetInBits();
    
    %pythoncode %{
        __swig_getmethods__["name"] = GetName
        if _newclass: x = property(GetName, None)
        
        __swig_getmethods__["type"] = GetType
        if _newclass: x = property(GetType, None)
        
        __swig_getmethods__["byte_offset"] = GetOffsetInBytes
        if _newclass: x = property(GetOffsetInBytes, None)
        
        __swig_getmethods__["bit_offset"] = GetOffsetInBits
        if _newclass: x = property(GetOffsetInBits, None)
    %}    

protected:
    std::auto_ptr<lldb_private::TypeMemberImpl> m_opaque_ap;
};

%feature("docstring",
"Represents a data type in lldb.  The FindFirstType() method of SBTarget/SBModule
returns a SBType.

SBType supports the eq/ne operator. For example,

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

int main (int argc, char const *argv[])
{
    Task *task_head = new Task(-1, NULL);
    Task *task1 = new Task(1, NULL);
    Task *task2 = new Task(2, NULL);
    Task *task3 = new Task(3, NULL); // Orphaned.
    Task *task4 = new Task(4, NULL);
    Task *task5 = new Task(5, NULL);

    task_head->next = task1;
    task1->next = task2;
    task2->next = task4;
    task4->next = task5;

    int total = 0;
    Task *t = task_head;
    while (t != NULL) {
        if (t->id >= 0)
            ++total;
        t = t->next;
    }
    printf('We have a total number of %d tasks\\n', total);

    // This corresponds to an empty task list.
    Task *empty_task_head = new Task(-1, NULL);

    return 0; // Break at this line
}

find_type.py:

        # Get the type 'Task'.
        task_type = target.FindFirstType('Task')
        self.assertTrue(task_type)

        # Get the variable 'task_head'.
        frame0.FindVariable('task_head')
        task_head_type = task_head.GetType()
        self.assertTrue(task_head_type.IsPointerType())

        # task_head_type is 'Task *'.
        task_pointer_type = task_type.GetPointerType()
        self.assertTrue(task_head_type == task_pointer_type)

        # Get the child mmember 'id' from 'task_head'.
        id = task_head.GetChildMemberWithName('id')
        id_type = id.GetType()

        # SBType.GetBasicType() takes an enum 'BasicType' (lldb-enumerations.h).
        int_type = id_type.GetBasicType(lldb.eBasicTypeInt)
        # id_type and int_type should be the same type!
        self.assertTrue(id_type == int_type)

...
") SBType;
class SBType
{
public:
    SBType (const lldb::SBType &rhs);

    ~SBType ();

    bool
    IsValid();

    size_t
    GetByteSize();

    bool
    IsPointerType();

    bool
    IsReferenceType();

    lldb::SBType
    GetPointerType();

    lldb::SBType
    GetPointeeType();

    lldb::SBType
    GetReferenceType();

    lldb::SBType
    GetDereferencedType();

    lldb::SBType
    GetUnqualifiedType();
    
    lldb::SBType
    GetBasicType (lldb::BasicType type);

    uint32_t
    GetNumberOfFields ();
    
    uint32_t
    GetNumberOfDirectBaseClasses ();
    
    uint32_t
    GetNumberOfVirtualBaseClasses ();
    
    lldb::SBTypeMember
    GetFieldAtIndex (uint32_t idx);
    
    lldb::SBTypeMember
    GetDirectBaseClassAtIndex (uint32_t idx);
    
    lldb::SBTypeMember
    GetVirtualBaseClassAtIndex (uint32_t idx);

    const char*
    GetName();
    
    lldb::TypeClass
    GetTypeClass ();
    
    %pythoncode %{
        __swig_getmethods__["name"] = GetName
        if _newclass: x = property(GetName, None)
        
        __swig_getmethods__["size"] = GetByteSize
        if _newclass: x = property(GetByteSize, None)
        
        __swig_getmethods__["is_pointer"] = IsPointerType
        if _newclass: x = property(IsPointerType, None)
        
        __swig_getmethods__["is_reference"] = IsReferenceType
        if _newclass: x = property(IsReferenceType, None)

        __swig_getmethods__["num_fields"] = GetNumberOfFields
        if _newclass: x = property(GetNumberOfFields, None)
        
        __swig_getmethods__["num_bases"] = GetNumberOfDirectBaseClasses
        if _newclass: x = property(GetNumberOfDirectBaseClasses, None)
        
        __swig_getmethods__["num_vbases"] = GetNumberOfVirtualBaseClasses
        if _newclass: x = property(GetNumberOfVirtualBaseClasses, None)
        
        __swig_getmethods__["class"] = GetTypeClass
        if _newclass: x = property(GetTypeClass, None)
    %}

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

    bool
    IsValid();

    void
    Append (lldb::SBType type);

    lldb::SBType
    GetTypeAtIndex (uint32_t index);

    uint32_t
    GetSize();

    ~SBTypeList();
};

} // namespace lldb

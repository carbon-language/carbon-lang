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
    printf('We have a total number of %d tasks\n', total);

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

    bool
    IsValid() const;

    void
    Append(const SBType& type);

    SBType
    GetTypeAtIndex(int index);

    int
    GetSize();

    ~SBTypeList();
};

} // namespace lldb

//===-- PythonDataObjects.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In order to guarantee correct working with Python, Python.h *MUST* be
// the *FIRST* header file included here.
#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#include "lldb/lldb-python.h"

#include <stdio.h>

#include "lldb/Core/Stream.h"
#include "lldb/Host/File.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

using namespace lldb_private;
using namespace lldb;

void
StructuredPythonObject::Dump(Stream &s) const
{
    s << "Python Obj: 0x" << GetValue();
}

//----------------------------------------------------------------------
// PythonObject
//----------------------------------------------------------------------

void
PythonObject::Dump (Stream &strm) const
{
    if (m_py_obj)
    {
        FILE *file = ::tmpfile();
        if (file)
        {
            ::PyObject_Print (m_py_obj, file, 0);
            const long length = ftell (file);
            if (length)
            {
                ::rewind(file);
                std::vector<char> file_contents (length,'\0');
                const size_t length_read = ::fread (file_contents.data(), 1, file_contents.size(), file);
                if (length_read > 0)
                    strm.Write (file_contents.data(), length_read);
            }
            ::fclose (file);
        }
    }
    else
        strm.PutCString ("NULL");
}

PyObjectType
PythonObject::GetObjectType() const
{
    if (IsNULLOrNone())
        return PyObjectType::None;

    if (PyList_Check(m_py_obj))
        return PyObjectType::List;
    if (PyDict_Check(m_py_obj))
        return PyObjectType::Dictionary;
    if (PyString_Check(m_py_obj))
        return PyObjectType::String;
    if (PyInt_Check(m_py_obj) || PyLong_Check(m_py_obj))
        return PyObjectType::Integer;
    return PyObjectType::Unknown;
}

PythonString
PythonObject::Repr ()
{
    if (!m_py_obj)
        return PythonString ();
    PyObject *repr = PyObject_Repr(m_py_obj);
    if (!repr)
        return PythonString ();
    return PythonString(repr);
}

PythonString
PythonObject::Str ()
{
    if (!m_py_obj)
        return PythonString ();
    PyObject *str = PyObject_Str(m_py_obj);
    if (!str)
        return PythonString ();
    return PythonString(str);
}

bool
PythonObject::IsNULLOrNone () const
{
    return ((m_py_obj == nullptr) || (m_py_obj == Py_None));
}

StructuredData::ObjectSP
PythonObject::CreateStructuredObject() const
{
    switch (GetObjectType())
    {
        case PyObjectType::Dictionary:
            return PythonDictionary(m_py_obj).CreateStructuredDictionary();
        case PyObjectType::Integer:
            return PythonInteger(m_py_obj).CreateStructuredInteger();
        case PyObjectType::List:
            return PythonList(m_py_obj).CreateStructuredArray();
        case PyObjectType::String:
            return PythonString(m_py_obj).CreateStructuredString();
        case PyObjectType::None:
            return StructuredData::ObjectSP();
        default:
            return StructuredData::ObjectSP(new StructuredPythonObject(m_py_obj));
    }
}

//----------------------------------------------------------------------
// PythonString
//----------------------------------------------------------------------

PythonString::PythonString (PyObject *py_obj) :
    PythonObject()
{
    Reset(py_obj); // Use "Reset()" to ensure that py_obj is a string
}

PythonString::PythonString (const PythonObject &object) :
    PythonObject()
{
    Reset(object.get()); // Use "Reset()" to ensure that py_obj is a string
}

PythonString::PythonString (llvm::StringRef string) :
    PythonObject(PyString_FromStringAndSize(string.data(), string.size()))
{
}

PythonString::PythonString(const char *string) :
    PythonObject(PyString_FromString(string))
{
}

PythonString::PythonString () :
    PythonObject()
{
}

PythonString::~PythonString ()
{
}

bool
PythonString::Reset (PyObject *py_obj)
{
    if (py_obj && PyString_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(nullptr);
    return py_obj == nullptr;
}

llvm::StringRef
PythonString::GetString() const
{
    if (m_py_obj)
        return llvm::StringRef(PyString_AsString(m_py_obj), GetSize());
    return llvm::StringRef();
}

size_t
PythonString::GetSize() const
{
    if (m_py_obj)
        return PyString_Size(m_py_obj);
    return 0;
}

void
PythonString::SetString (llvm::StringRef string)
{
    PythonObject::Reset(PyString_FromStringAndSize(string.data(), string.size()));
}

StructuredData::StringSP
PythonString::CreateStructuredString() const
{
    StructuredData::StringSP result(new StructuredData::String);
    result->SetValue(GetString());
    return result;
}

//----------------------------------------------------------------------
// PythonInteger
//----------------------------------------------------------------------

PythonInteger::PythonInteger (PyObject *py_obj) :
    PythonObject()
{
    Reset(py_obj); // Use "Reset()" to ensure that py_obj is a integer type
}

PythonInteger::PythonInteger (const PythonObject &object) :
    PythonObject()
{
    Reset(object.get()); // Use "Reset()" to ensure that py_obj is a integer type
}

PythonInteger::PythonInteger (int64_t value) :
    PythonObject()
{
    SetInteger (value);
}


PythonInteger::~PythonInteger ()
{
}

bool
PythonInteger::Reset (PyObject *py_obj)
{
    if (py_obj)
    {
        if (PyInt_Check (py_obj) || PyLong_Check(py_obj))
            return PythonObject::Reset(py_obj);
    }
    
    PythonObject::Reset(nullptr);
    return py_obj == nullptr;
}

int64_t
PythonInteger::GetInteger() const
{
    if (m_py_obj)
    {
        if (PyInt_Check(m_py_obj))
            return PyInt_AsLong(m_py_obj);
        else if (PyLong_Check(m_py_obj))
            return PyLong_AsLongLong(m_py_obj);
    }
    return UINT64_MAX;
}

void
PythonInteger::SetInteger (int64_t value)
{
    PythonObject::Reset(PyLong_FromLongLong(value));
}

StructuredData::IntegerSP
PythonInteger::CreateStructuredInteger() const
{
    StructuredData::IntegerSP result(new StructuredData::Integer);
    result->SetValue(GetInteger());
    return result;
}

//----------------------------------------------------------------------
// PythonList
//----------------------------------------------------------------------

PythonList::PythonList (bool create_empty) :
    PythonObject(create_empty ? PyList_New(0) : nullptr)
{
}

PythonList::PythonList (uint32_t count) :
    PythonObject(PyList_New(count))
{
}

PythonList::PythonList (PyObject *py_obj) :
    PythonObject()
{
    Reset(py_obj); // Use "Reset()" to ensure that py_obj is a list
}


PythonList::PythonList (const PythonObject &object) :
    PythonObject()
{
    Reset(object.get()); // Use "Reset()" to ensure that py_obj is a list
}

PythonList::~PythonList ()
{
}

bool
PythonList::Reset (PyObject *py_obj)
{
    if (py_obj && PyList_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(nullptr);
    return py_obj == nullptr;
}

uint32_t
PythonList::GetSize() const
{
    if (m_py_obj)
        return PyList_GET_SIZE(m_py_obj);
    return 0;
}

PythonObject
PythonList::GetItemAtIndex(uint32_t index) const
{
    if (m_py_obj)
        return PythonObject(PyList_GetItem(m_py_obj, index));
    return PythonObject();
}

void
PythonList::SetItemAtIndex (uint32_t index, const PythonObject & object)
{
    if (m_py_obj && object)
        PyList_SetItem(m_py_obj, index, object.get());
}

void
PythonList::AppendItem (const PythonObject &object)
{
    if (m_py_obj && object)
        PyList_Append(m_py_obj, object.get());
}

StructuredData::ArraySP
PythonList::CreateStructuredArray() const
{
    StructuredData::ArraySP result(new StructuredData::Array);
    uint32_t count = GetSize();
    for (uint32_t i = 0; i < count; ++i)
    {
        PythonObject obj = GetItemAtIndex(i);
        result->AddItem(obj.CreateStructuredObject());
    }
    return result;
}

//----------------------------------------------------------------------
// PythonDictionary
//----------------------------------------------------------------------

PythonDictionary::PythonDictionary (bool create_empty) :
PythonObject(create_empty ? PyDict_New() : nullptr)
{
}

PythonDictionary::PythonDictionary (PyObject *py_obj) :
    PythonObject(py_obj)
{
    Reset(py_obj); // Use "Reset()" to ensure that py_obj is a dictionary
}


PythonDictionary::PythonDictionary (const PythonObject &object) :
    PythonObject()
{
    Reset(object.get()); // Use "Reset()" to ensure that py_obj is a dictionary
}

PythonDictionary::~PythonDictionary ()
{
}

bool
PythonDictionary::Reset (PyObject *py_obj)
{
    if (py_obj && PyDict_Check(py_obj))
        return PythonObject::Reset(py_obj);
    
    PythonObject::Reset(nullptr);
    return py_obj == nullptr;
}

uint32_t
PythonDictionary::GetSize() const
{
    if (m_py_obj)
        return PyDict_Size(m_py_obj);
    return 0;
}

PythonObject
PythonDictionary::GetItemForKey (const char *key) const
{
    if (key && key[0])
    {
        PythonString python_key(key);
        return GetItemForKey(python_key);
    }
    return PythonObject();
}


PythonObject
PythonDictionary::GetItemForKey (const PythonString &key) const
{
    if (m_py_obj && key)
        return PythonObject(PyDict_GetItem(m_py_obj, key.get()));
    return PythonObject();
}


const char *
PythonDictionary::GetItemForKeyAsString (const PythonString &key, const char *fail_value) const
{
    if (m_py_obj && key)
    {
        PyObject *py_obj = PyDict_GetItem(m_py_obj, key.get());
        if (py_obj && PyString_Check(py_obj))
            return PyString_AsString(py_obj);
    }
    return fail_value;
}

int64_t
PythonDictionary::GetItemForKeyAsInteger (const PythonString &key, int64_t fail_value) const
{
    if (m_py_obj && key)
    {
        PyObject *py_obj = PyDict_GetItem(m_py_obj, key.get());
        if (py_obj)
        {
            if (PyInt_Check(py_obj))
                return PyInt_AsLong(py_obj);

            if (PyLong_Check(py_obj))
                return PyLong_AsLong(py_obj);
        }
    }
    return fail_value;
}

PythonList
PythonDictionary::GetKeys () const
{
    if (m_py_obj)
        return PythonList(PyDict_Keys(m_py_obj));
    return PythonList(true);
}

PythonString
PythonDictionary::GetKeyAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (m_py_obj)
    {
        while (PyDict_Next(m_py_obj, &pos_iter, &key, &value))
        {
            if (pos-- == 0)
                return PythonString(key);
        }
    }
    return PythonString();
}

PythonObject
PythonDictionary::GetValueAtPosition (uint32_t pos) const
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (!m_py_obj)
        return PythonObject();
    
    while (PyDict_Next(m_py_obj, &pos_iter, &key, &value)) {
        if (pos-- == 0)
            return PythonObject(value);
    }
    return PythonObject();
}

void
PythonDictionary::SetItemForKey (const PythonString &key, PyObject *value)
{
    if (m_py_obj && key && value)
        PyDict_SetItem(m_py_obj, key.get(), value);
}

void
PythonDictionary::SetItemForKey (const PythonString &key, const PythonObject &value)
{
    if (m_py_obj && key && value)
        PyDict_SetItem(m_py_obj, key.get(), value.get());
}

StructuredData::DictionarySP
PythonDictionary::CreateStructuredDictionary() const
{
    StructuredData::DictionarySP result(new StructuredData::Dictionary);
    PythonList keys(GetKeys());
    uint32_t num_keys = keys.GetSize();
    for (uint32_t i = 0; i < num_keys; ++i)
    {
        PythonObject key = keys.GetItemAtIndex(i);
        PythonString key_str = key.Str();
        PythonObject value = GetItemForKey(key);
        StructuredData::ObjectSP structured_value = value.CreateStructuredObject();
        result->AddItem(key_str.GetString(), structured_value);
    }
    return result;
}

#endif

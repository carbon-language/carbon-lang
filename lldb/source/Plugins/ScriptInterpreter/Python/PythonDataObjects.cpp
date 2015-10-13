//===-- PythonDataObjects.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#include "lldb-python.h"
#include "PythonDataObjects.h"
#include "ScriptInterpreterPython.h"

#include "lldb/Core/Stream.h"
#include "lldb/Host/File.h"
#include "lldb/Interpreter/ScriptInterpreter.h"

#include <stdio.h>

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
PythonObject::Dump(Stream &strm) const
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
    if (!IsAllocated())
        return PyObjectType::None;

    if (PyList_Check(m_py_obj))
        return PyObjectType::List;
    if (PyDict_Check(m_py_obj))
        return PyObjectType::Dictionary;
    if (PyUnicode_Check(m_py_obj))
        return PyObjectType::String;
    if (PyLong_Check(m_py_obj))
        return PyObjectType::Integer;
#if PY_MAJOR_VERSION < 3
    // These functions don't exist in Python 3.x.  PyString is PyUnicode
    // and PyInt is PyLong.
    if (PyString_Check(m_py_obj))
        return PyObjectType::String;
    if (PyInt_Check(m_py_obj))
        return PyObjectType::Integer;
#endif
    return PyObjectType::Unknown;
}

PythonString
PythonObject::Repr()
{
    if (!m_py_obj)
        return PythonString();
    PyObject *repr = PyObject_Repr(m_py_obj);
    if (!repr)
        return PythonString();
    return PythonString(PyRefType::Owned, repr);
}

PythonString
PythonObject::Str()
{
    if (!m_py_obj)
        return PythonString();
    PyObject *str = PyObject_Str(m_py_obj);
    if (!str)
        return PythonString();
    return PythonString(PyRefType::Owned, str);
}

bool
PythonObject::IsNone() const
{
    return m_py_obj == Py_None;
}

bool
PythonObject::IsValid() const
{
    return m_py_obj != nullptr;
}

bool
PythonObject::IsAllocated() const
{
    return IsValid() && !IsNone();
}

StructuredData::ObjectSP
PythonObject::CreateStructuredObject() const
{
    switch (GetObjectType())
    {
        case PyObjectType::Dictionary:
            return PythonDictionary(PyRefType::Borrowed, m_py_obj).CreateStructuredDictionary();
        case PyObjectType::Integer:
            return PythonInteger(PyRefType::Borrowed, m_py_obj).CreateStructuredInteger();
        case PyObjectType::List:
            return PythonList(PyRefType::Borrowed, m_py_obj).CreateStructuredArray();
        case PyObjectType::String:
            return PythonString(PyRefType::Borrowed, m_py_obj).CreateStructuredString();
        case PyObjectType::None:
            return StructuredData::ObjectSP();
        default:
            return StructuredData::ObjectSP(new StructuredPythonObject(m_py_obj));
    }
}

//----------------------------------------------------------------------
// PythonString
//----------------------------------------------------------------------

PythonString::PythonString(PyRefType type, PyObject *py_obj)
    : PythonObject()
{
    Reset(type, py_obj); // Use "Reset()" to ensure that py_obj is a string
}

PythonString::PythonString(const PythonString &object)
    : PythonObject(object)
{
}

PythonString::PythonString(llvm::StringRef string)
    : PythonObject()
{
    SetString(string);
}

PythonString::PythonString(const char *string)
    : PythonObject()
{
    SetString(llvm::StringRef(string));
}

PythonString::PythonString()
    : PythonObject()
{
}

PythonString::~PythonString ()
{
}

bool
PythonString::Check(PyObject *py_obj)
{
    if (!py_obj)
        return false;
#if PY_MAJOR_VERSION >= 3
    // Python 3 does not have PyString objects, only PyUnicode.
    return PyUnicode_Check(py_obj);
#else
    return PyUnicode_Check(py_obj) || PyString_Check(py_obj);
#endif
}

void
PythonString::Reset(PyRefType type, PyObject *py_obj)
{
    // Grab the desired reference type so that if we end up rejecting
    // `py_obj` it still gets decremented if necessary.
    PythonObject result(type, py_obj);

    if (!PythonString::Check(py_obj))
    {
        PythonObject::Reset();
        return;
    }

    // Convert this to a PyBytes object, and only store the PyBytes.  Note that in
    // Python 2.x, PyString and PyUnicode are interchangeable, and PyBytes is an alias
    // of PyString.  So on 2.x, if we get into this branch, we already have a PyBytes.
    if (PyUnicode_Check(py_obj))
    {
        // Since we're converting this to a different object, we assume ownership of the
        // new object regardless of the value of `type`.
        result.Reset(PyRefType::Owned, PyUnicode_AsUTF8String(py_obj));
    }

    assert(PyBytes_Check(result.get()) && "PythonString::Reset received a non-string");

    // Calling PythonObject::Reset(const PythonObject&) will lead to stack overflow since it calls
    // back into the virtual implementation.
    PythonObject::Reset(PyRefType::Borrowed, result.get());
}

llvm::StringRef
PythonString::GetString() const
{
    if (IsValid())
    {
        Py_ssize_t size;
        char *c;
        PyBytes_AsStringAndSize(m_py_obj, &c, &size);
        return llvm::StringRef(c, size);
    }
    return llvm::StringRef();
}

size_t
PythonString::GetSize() const
{
    if (IsValid())
        return PyBytes_Size(m_py_obj);
    return 0;
}

void
PythonString::SetString (llvm::StringRef string)
{
#if PY_MAJOR_VERSION >= 3
    PyObject *unicode = PyUnicode_FromStringAndSize(string.data(), string.size());
    PyObject *bytes = PyUnicode_AsUTF8String(unicode);
    PythonObject::Reset(PyRefType::Owned, bytes);
    Py_DECREF(unicode);
#else
    PythonObject::Reset(PyRefType::Owned, PyString_FromStringAndSize(string.data(), string.size()));
#endif
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

PythonInteger::PythonInteger(PyRefType type, PyObject *py_obj)
    : PythonObject()
{
    Reset(type, py_obj); // Use "Reset()" to ensure that py_obj is a integer type
}

PythonInteger::PythonInteger(const PythonInteger &object)
    : PythonObject(object)
{
}

PythonInteger::PythonInteger(int64_t value)
    : PythonObject()
{
    SetInteger(value);
}


PythonInteger::~PythonInteger ()
{
}

bool
PythonInteger::Check(PyObject *py_obj)
{
    if (!py_obj)
        return false;

#if PY_MAJOR_VERSION >= 3
    // Python 3 does not have PyInt_Check.  There is only one type of
    // integral value, long.
    return PyLong_Check(py_obj);
#else
    return PyLong_Check(py_obj) || PyInt_Check(py_obj);
#endif
}

void
PythonInteger::Reset(PyRefType type, PyObject *py_obj)
{
    // Grab the desired reference type so that if we end up rejecting
    // `py_obj` it still gets decremented if necessary.
    PythonObject result(type, py_obj);

    if (!PythonInteger::Check(py_obj))
    {
        PythonObject::Reset();
        return;
    }

#if PY_MAJOR_VERSION < 3
    // Always store this as a PyLong, which makes interoperability between
    // Python 2.x and Python 3.x easier.  This is only necessary in 2.x,
    // since 3.x doesn't even have a PyInt.
    if (PyInt_Check(py_obj))
    {
        // Since we converted the original object to a different type, the new
        // object is an owned object regardless of the ownership semantics requested
        // by the user.
        result.Reset(PyRefType::Owned, PyLong_FromLongLong(PyInt_AsLong(py_obj)));
    }
#endif

    assert(PyLong_Check(result.get()) && "Couldn't get a PyLong from this PyObject");

    // Calling PythonObject::Reset(const PythonObject&) will lead to stack overflow since it calls
    // back into the virtual implementation.
    PythonObject::Reset(PyRefType::Borrowed, result.get());
}

int64_t
PythonInteger::GetInteger() const
{
    if (m_py_obj)
    {
        assert(PyLong_Check(m_py_obj) && "PythonInteger::GetInteger has a PyObject that isn't a PyLong");

        return PyLong_AsLongLong(m_py_obj);
    }
    return UINT64_MAX;
}

void
PythonInteger::SetInteger(int64_t value)
{
    PythonObject::Reset(PyRefType::Owned, PyLong_FromLongLong(value));
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

PythonList::PythonList(PyInitialValue value)
    : PythonObject()
{
    if (value == PyInitialValue::Empty)
        Reset(PyRefType::Owned, PyList_New(0));
}

PythonList::PythonList(PyRefType type, PyObject *py_obj)
    : PythonObject()
{
    Reset(type, py_obj); // Use "Reset()" to ensure that py_obj is a list
}

PythonList::PythonList(const PythonList &list)
    : PythonObject(list)
{
}

PythonList::~PythonList ()
{
}

bool
PythonList::Check(PyObject *py_obj)
{
    if (!py_obj)
        return false;
    return PyList_Check(py_obj);
}

void
PythonList::Reset(PyRefType type, PyObject *py_obj)
{
    // Grab the desired reference type so that if we end up rejecting
    // `py_obj` it still gets decremented if necessary.
    PythonObject result(type, py_obj);

    if (!PythonList::Check(py_obj))
    {
        PythonObject::Reset();
        return;
    }

    // Calling PythonObject::Reset(const PythonObject&) will lead to stack overflow since it calls
    // back into the virtual implementation.
    PythonObject::Reset(PyRefType::Borrowed, result.get());
}

uint32_t
PythonList::GetSize() const
{
    if (IsValid())
        return PyList_GET_SIZE(m_py_obj);
    return 0;
}

PythonObject
PythonList::GetItemAtIndex(uint32_t index) const
{
    if (IsValid())
        return PythonObject(PyRefType::Borrowed, PyList_GetItem(m_py_obj, index));
    return PythonObject();
}

void
PythonList::SetItemAtIndex(uint32_t index, const PythonObject &object)
{
    if (IsAllocated() && object.IsValid())
    {
        // PyList_SetItem is documented to "steal" a reference, so we need to
        // convert it to an owned reference by incrementing it.
        Py_INCREF(object.get());
        PyList_SetItem(m_py_obj, index, object.get());
    }
}

void
PythonList::AppendItem(const PythonObject &object)
{
    if (IsAllocated() && object.IsValid())
    {
        // `PyList_Append` does *not* steal a reference, so do not call `Py_INCREF`
        // here like we do with `PyList_SetItem`.
        PyList_Append(m_py_obj, object.get());
    }
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

PythonDictionary::PythonDictionary(PyInitialValue value)
    : PythonObject()
{
    if (value == PyInitialValue::Empty)
        Reset(PyRefType::Owned, PyDict_New());
}

PythonDictionary::PythonDictionary(PyRefType type, PyObject *py_obj)
    : PythonObject()
{
    Reset(type, py_obj); // Use "Reset()" to ensure that py_obj is a dictionary
}

PythonDictionary::PythonDictionary(const PythonDictionary &object)
    : PythonObject(object)
{
}

PythonDictionary::~PythonDictionary ()
{
}

bool
PythonDictionary::Check(PyObject *py_obj)
{
    if (!py_obj)
        return false;

    return PyDict_Check(py_obj);
}

void
PythonDictionary::Reset(PyRefType type, PyObject *py_obj)
{
    // Grab the desired reference type so that if we end up rejecting
    // `py_obj` it still gets decremented if necessary.
    PythonObject result(type, py_obj);

    if (!PythonDictionary::Check(py_obj))
    {
        PythonObject::Reset();
        return;
    }

    // Calling PythonObject::Reset(const PythonObject&) will lead to stack overflow since it calls
    // back into the virtual implementation.
    PythonObject::Reset(PyRefType::Borrowed, result.get());
}

uint32_t
PythonDictionary::GetSize() const
{
    if (IsValid())
        return PyDict_Size(m_py_obj);
    return 0;
}

PythonList
PythonDictionary::GetKeys() const
{
    if (IsValid())
        return PythonList(PyRefType::Owned, PyDict_Keys(m_py_obj));
    return PythonList(PyInitialValue::Invalid);
}

PythonObject
PythonDictionary::GetItemForKey(const PythonObject &key) const
{
    if (IsAllocated() && key.IsValid())
        return PythonObject(PyRefType::Borrowed, PyDict_GetItem(m_py_obj, key.get()));
    return PythonObject();
}

void
PythonDictionary::SetItemForKey(const PythonObject &key, const PythonObject &value)
{
    if (IsAllocated() && key.IsValid() && value.IsValid())
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
        PythonObject value = GetItemForKey(key);
        StructuredData::ObjectSP structured_value = value.CreateStructuredObject();
        result->AddItem(key.Str().GetString(), structured_value);
    }
    return result;
}

#endif

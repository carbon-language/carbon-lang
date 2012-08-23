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

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include "PythonDataObjects.h"

using namespace lldb_private;
using namespace lldb;

PythonDataObject::PythonDataObject (PyObject* object) : m_object(object)
{
}

PythonDataString*
PythonDataObject::GetStringObject ()
{
    return new PythonDataString(GetPythonObject());
}
    
PythonDataInteger*
PythonDataObject::GetIntegerObject ()
{
    return new PythonDataInteger(GetPythonObject());
}

PythonDataArray*
PythonDataObject::GetArrayObject()
{
    return new PythonDataArray(GetPythonObject());
}

PythonDataDictionary*
PythonDataObject::GetDictionaryObject()
{
    return new PythonDataDictionary(GetPythonObject());
}

PythonDataInteger::PythonDataInteger (PyObject* object) : m_object(object)
{
    if (!PyInt_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataInteger::~PythonDataInteger ()
{
}

PythonDataInteger::PythonDataInteger (int64_t value) : m_object(PyInt_FromLong(value))
{
}

int64_t
PythonDataInteger::GetInteger()
{
    if (m_object)
        return PyInt_AsLong(GetPythonObject());
    else
        return UINT64_MAX;
}

void
PythonDataInteger::SetInteger (int64_t value)
{
    m_object.Reset(PyInt_FromLong(value));
}

PythonDataString::PythonDataString (PyObject* object) : m_object(object)
{
    if (!PyString_Check(GetPythonObject()))
        m_object.Reset();}

PythonDataString::PythonDataString (const char* string) : m_object(PyString_FromString(string))
{
}

PythonDataString::~PythonDataString ()
{
}

const char*
PythonDataString::GetString()
{
    if (m_object)
        return PyString_AsString(GetPythonObject());
    return NULL;
}

void
PythonDataString::SetString (const char* string)
{
    m_object.Reset(PyString_FromString(string));
}

PythonDataArray::PythonDataArray (uint32_t count) : m_object(PyList_New(count))
{
}

PythonDataArray::PythonDataArray (PyObject* object) : m_object(object)
{
    if (!PyList_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataArray::~PythonDataArray ()
{
}

uint32_t
PythonDataArray::GetSize()
{
    if (m_object)
        return PyList_GET_SIZE(GetPythonObject());
    return 0;
}

PythonDataObject*
PythonDataArray::GetItemAtIndex (uint32_t index)
{
    if (m_object)
        return new PythonDataObject(PyList_GetItem(GetPythonObject(), index));
    return NULL;
}

void
PythonDataArray::SetItemAtIndex (uint32_t index, PythonDataObject* object)
{
    if (m_object && object && *object)
        PyList_SetItem(GetPythonObject(), index, object->GetPythonObject());
}

void
PythonDataArray::AppendItem (PythonDataObject* object)
{
    if (m_object && object && *object)
        PyList_Append(GetPythonObject(), object->GetPythonObject());
}

PythonDataDictionary::PythonDataDictionary () : m_object(PyDict_New())
{
}

PythonDataDictionary::PythonDataDictionary (PyObject* object) : m_object(object)
{
    if (!PyDict_Check(GetPythonObject()))
        m_object.Reset();
}

PythonDataDictionary::~PythonDataDictionary ()
{
}

uint32_t
PythonDataDictionary::GetSize()
{
    if (m_object)
        return PyDict_Size(GetPythonObject());
    return 0;
}

PythonDataObject*
PythonDataDictionary::GetItemForKey (PythonDataString* key)
{
    if (m_object && key && *key)
        return new PythonDataObject(PyDict_GetItem(GetPythonObject(), key->GetPythonObject()));
    return NULL;
}

PythonDataArray*
PythonDataDictionary::GetKeys ()
{
    if (m_object)
        return new PythonDataArray(PyDict_Keys(GetPythonObject()));
    return NULL;
}

PythonDataString*
PythonDataDictionary::GetKeyAtPosition (uint32_t pos)
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (!m_object)
        return NULL;
    
    while (PyDict_Next(GetPythonObject(), &pos_iter, &key, &value)) {
        if (pos-- == 0)
            return new PythonDataString(key);
    }
    return NULL;
}

PythonDataObject*
PythonDataDictionary::GetValueAtPosition (uint32_t pos)
{
    PyObject *key, *value;
    Py_ssize_t pos_iter = 0;
    
    if (!m_object)
        return NULL;
    
    while (PyDict_Next(GetPythonObject(), &pos_iter, &key, &value)) {
        if (pos-- == 0)
            return new PythonDataObject(value);
    }
    return NULL;
}

void
PythonDataDictionary::SetItemForKey (PythonDataString* key, PythonDataObject* value)
{
    if (m_object && key && value && *key && *value)
        PyDict_SetItem(GetPythonObject(), key->GetPythonObject(), value->GetPythonObject());
}

#endif

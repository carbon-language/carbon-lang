//===-- PythonDataObjects.h----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PythonDataObjects_h_
#define liblldb_PythonDataObjects_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-defines.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Flags.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {
    
    class PythonRefCountedObject
    {
    public:
        PythonRefCountedObject (PyObject* obj) : m_object(obj)
        {
            Py_XINCREF(m_object);
        }
        
        ~PythonRefCountedObject ()
        {
            Py_XDECREF(m_object);
        }
        
        void
        Reset (PyObject* object = NULL)
        {
            Py_XDECREF(m_object);
            m_object = object;
            Py_XINCREF(m_object);
        }
        
        PyObject*
        GetPyhonObject ()
        {
            return m_object;
        }
        
        operator bool ()
        {
            return m_object != NULL;
        }
        
    private:
        PyObject* m_object;
    };
    
    class PythonDataString
    {
    public:
        
        PythonDataString (PyObject* object);
        PythonDataString (const char* string);
        ~PythonDataString ();
        
        const char*
        GetString();
        
        void
        SetString (const char* string);
        
        operator bool ()
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() { return m_object.GetPyhonObject(); }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataInteger
    {
    public:
        
        PythonDataInteger (PyObject* object);
        PythonDataInteger (int64_t value);
        ~PythonDataInteger ();
        
        int64_t
        GetInteger();
        
        void
        SetInteger (int64_t value);
        
        operator bool ()
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() { return m_object.GetPyhonObject(); }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataArray
    {
    public:
        
        PythonDataArray (uint32_t count);
        PythonDataArray (PyObject* object);
        ~PythonDataArray ();
        
        uint32_t
        GetSize();
        
        PythonDataObject*
        GetItemAtIndex (uint32_t index);
        
        void
        SetItemAtIndex (uint32_t index, PythonDataObject* object);
        
        void
        AppendItem (PythonDataObject* object);
        
        operator bool ()
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() { return m_object.GetPyhonObject(); }
    private:
        PythonRefCountedObject m_object;
    };
    
    class PythonDataDictionary
    {
    public:
        
        PythonDataDictionary ();
        PythonDataDictionary (PyObject* object);
        ~PythonDataDictionary ();
        
        uint32_t GetSize();
        
        PythonDataObject*
        GetItemForKey (PythonDataString* key);
        
        typedef bool (*DictionaryIteratorCallback)(PythonDataString* key, PythonDataDictionary* dict);
        
        PythonDataArray*
        GetKeys ();
        
        PythonDataString*
        GetKeyAtPosition (uint32_t pos);
        
        PythonDataObject*
        GetValueAtPosition (uint32_t pos);
        
        void
        SetItemForKey (PythonDataString* key, PythonDataObject* value);
        
        operator bool ()
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() { return m_object.GetPyhonObject(); }
    private:
        PythonRefCountedObject m_object;
    };

    class PythonDataObject
    {
    public:
        
        PythonDataObject (PyObject* object);
        
        ~PythonDataObject ();
        
        PythonDataString*
        GetStringObject ();
        
        PythonDataInteger*
        GetIntegerObject ();
        
        PythonDataArray*
        GetArrayObject();
        
        PythonDataDictionary*
        GetDictionaryObject();
        
        operator bool ()
        {
            return m_object.operator bool();
        }
        
        PyObject*
        GetPythonObject() { return m_object.GetPyhonObject(); }
        
    private:
        PythonRefCountedObject m_object;
    };
    
} // namespace lldb_private

#endif  // liblldb_PythonDataObjects_h_

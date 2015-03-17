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
#include "lldb/Core/StructuredData.h"
#include "lldb/Core/Flags.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/lldb-python.h"

namespace lldb_private {
class PythonString;
class PythonList;
class PythonDictionary;
class PythonObject;
class PythonInteger;

class StructuredPythonObject : public StructuredData::Generic
{
  public:
    StructuredPythonObject()
        : StructuredData::Generic()
    {
    }

    StructuredPythonObject(void *obj)
        : StructuredData::Generic(obj)
    {
        Py_XINCREF(GetValue());
    }

    virtual ~StructuredPythonObject()
    {
        if (Py_IsInitialized())
            Py_XDECREF(GetValue());
        SetValue(nullptr);
    }

    bool
    IsValid() const override
    {
        return GetValue() && GetValue() != Py_None;
    }

    void Dump(Stream &s) const override;

  private:
    DISALLOW_COPY_AND_ASSIGN(StructuredPythonObject);
};

enum class PyObjectType
{
    Unknown,
    None,
    Integer,
    Dictionary,
    List,
    String
};

    class PythonObject
    {
    public:
        PythonObject () :
            m_py_obj(NULL)
        {
        }
        
        explicit PythonObject (PyObject* py_obj) :
            m_py_obj(NULL)
        {
            Reset (py_obj);
        }
        
        PythonObject (const PythonObject &rhs) :
            m_py_obj(NULL)
        {
            Reset (rhs.m_py_obj);
        }

        virtual
        ~PythonObject ()
        {
            Reset (NULL);
        }

        bool
        Reset (const PythonObject &object)
        {
            return Reset(object.get());
        }

        virtual bool
        Reset (PyObject* py_obj = NULL)
        {
            if (py_obj != m_py_obj)
            {
                if (Py_IsInitialized())
                    Py_XDECREF(m_py_obj);
                m_py_obj = py_obj;
                if (Py_IsInitialized())
                    Py_XINCREF(m_py_obj);
            }
            return true;
        }
        
        void
        Dump () const
        {
            if (m_py_obj)
                _PyObject_Dump (m_py_obj);
            else
                puts ("NULL");
        }
        
        void
        Dump (Stream &strm) const;

        PyObject*
        get () const
        {
            return m_py_obj;
        }

        PyObjectType GetObjectType() const;

        PythonString
        Repr ();
        
        PythonString
        Str ();
        
        explicit operator bool () const
        {
            return m_py_obj != NULL;
        }
        
        bool
        IsNULLOrNone () const;

        StructuredData::ObjectSP CreateStructuredObject() const;

    protected:
        PyObject* m_py_obj;
    };
    
    class PythonString: public PythonObject
    {
    public:
        PythonString ();
        PythonString (PyObject *o);
        PythonString (const PythonObject &object);
        PythonString (llvm::StringRef string);
        PythonString (const char *string);
        virtual ~PythonString ();

        virtual bool
        Reset (PyObject* py_obj = NULL);

        llvm::StringRef
        GetString() const;

        size_t
        GetSize() const;

        void SetString(llvm::StringRef string);

        StructuredData::StringSP CreateStructuredString() const;
    };
    
    class PythonInteger: public PythonObject
    {
    public:
        
        PythonInteger ();
        PythonInteger (PyObject* py_obj);
        PythonInteger (const PythonObject &object);
        PythonInteger (int64_t value);
        virtual ~PythonInteger ();
        
        virtual bool
        Reset (PyObject* py_obj = NULL);

        int64_t GetInteger() const;

        void
        SetInteger (int64_t value);

        StructuredData::IntegerSP CreateStructuredInteger() const;
    };
    
    class PythonList: public PythonObject
    {
    public:
        
        PythonList (bool create_empty);
        PythonList (PyObject* py_obj);
        PythonList (const PythonObject &object);
        PythonList (uint32_t count);
        virtual ~PythonList ();
        
        virtual bool
        Reset (PyObject* py_obj = NULL);

        uint32_t GetSize() const;

        PythonObject GetItemAtIndex(uint32_t index) const;

        void
        SetItemAtIndex (uint32_t index, const PythonObject &object);
        
        void
        AppendItem (const PythonObject &object);

        StructuredData::ArraySP CreateStructuredArray() const;
    };
    
    class PythonDictionary: public PythonObject
    {
    public:
        
        explicit PythonDictionary (bool create_empty);
        PythonDictionary (PyObject* object);
        PythonDictionary (const PythonObject &object);
        virtual ~PythonDictionary ();
        
        virtual bool
        Reset (PyObject* object = NULL);

        uint32_t GetSize() const;

        PythonObject
        GetItemForKey (const PythonString &key) const;
        
        const char *
        GetItemForKeyAsString (const PythonString &key, const char *fail_value = NULL) const;

        int64_t
        GetItemForKeyAsInteger (const PythonString &key, int64_t fail_value = 0) const;

        PythonObject
        GetItemForKey (const char *key) const;

        typedef bool (*DictionaryIteratorCallback)(PythonString* key, PythonDictionary* dict);
        
        PythonList
        GetKeys () const;
        
        PythonString
        GetKeyAtPosition (uint32_t pos) const;
        
        PythonObject
        GetValueAtPosition (uint32_t pos) const;
        
        void
        SetItemForKey (const PythonString &key, PyObject *value);

        void
        SetItemForKey (const PythonString &key, const PythonObject& value);

        StructuredData::DictionarySP CreateStructuredDictionary() const;
    };
    
} // namespace lldb_private

#endif  // liblldb_PythonDataObjects_h_

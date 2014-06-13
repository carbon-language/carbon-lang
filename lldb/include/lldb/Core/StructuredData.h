//===-- StructuredData.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StructuredData_h_
#define liblldb_StructuredData_h_

// C Includes
// C++ Includes

#include <map>
#include <utility>
#include <vector>
#include <string>

#include "llvm/ADT/StringRef.h"

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-defines.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class StructuredData StructuredData.h "lldb/Core/StructuredData.h"
/// @brief A class which can hold structured data
///
/// The StructuredData class is designed to hold the data from a JSON
/// or plist style file -- a serialized data structure with dictionaries 
/// (maps, hashes), arrays, and concrete values like integers, floating 
/// point numbers, strings, booleans.
///
/// StructuredData does not presuppose any knowledge of the schema for
/// the data it is holding; it can parse JSON data, for instance, and
/// other parts of lldb can iterate through the parsed data set to find
/// keys and values that may be present.  
//----------------------------------------------------------------------

class StructuredData
{
public:

    class Object;
    class Array;
    class Integer;
    class Float;
    class Boolean;
    class String;
    class Dictionary;

    typedef std::shared_ptr<Object> ObjectSP;
    typedef std::shared_ptr<Array> ArraySP;
    typedef std::shared_ptr<Dictionary> DictionarySP;

    enum class Type {
        eTypeInvalid = -1,
        eTypeNull = 0,
        eTypeArray,
        eTypeInteger,
        eTypeFloat,
        eTypeBoolean,
        eTypeString,
        eTypeDictionary
    };

    class Object :
        public std::enable_shared_from_this<Object>
    {
    public:

        Object (Type t = Type::eTypeInvalid) :
            m_type (t)
        {
        }

        virtual ~Object ()
        {
        }

        virtual void
        Clear ()
        {
            m_type = Type::eTypeInvalid;
        }

        Type
        GetType () const
        {
            return m_type;
        }

        void
        SetType (Type t)
        {
            m_type = t;
        }

        Array *
        GetAsArray ()
        {
            if (m_type == Type::eTypeArray)
                return (Array *)this;
            return NULL;
        }

        Dictionary *
        GetAsDictionary ()
        {
            if (m_type == Type::eTypeDictionary)
                return (Dictionary *)this;
            return NULL;
        }

        Integer *
        GetAsInteger ()
        {
            if (m_type == Type::eTypeInteger)
                return (Integer *)this;
            return NULL;
        }

        Float *
        GetAsFloat ()
        {
            if (m_type == Type::eTypeFloat)
                return (Float *)this;
            return NULL;
        }

        Boolean *
        GetAsBoolean ()
        {
            if (m_type == Type::eTypeBoolean)
                return (Boolean *)this;
            return NULL;
        }

        String *
        GetAsString ()
        {
            if (m_type == Type::eTypeString)
                return (String *)this;
            return NULL;
        }

        ObjectSP
        GetObjectForDotSeparatedPath (llvm::StringRef path);

        virtual void
        Dump (Stream &s) const = 0; 

    private:
        Type m_type;
    };

    class Array : public Object
    {
    public:
        Array () :
            Object (Type::eTypeArray)
        {
        }

        virtual
        ~Array()
        {
        }

        size_t
        GetSize()
        {
            return m_items.size();
        }

        ObjectSP
        operator[](size_t idx)
        {
            if (idx < m_items.size())
                return m_items[idx];
            return ObjectSP();
        }

        ObjectSP
        GetItemAtIndex (size_t idx)
        {
            if (idx < m_items.size())
                return m_items[idx];
            return ObjectSP();
        }

        void
        Push(ObjectSP item)
        {
            m_items.push_back(item);
        }

        void
        AddItem(ObjectSP item)
        {
            m_items.push_back(item);
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        typedef std::vector<ObjectSP> collection;
        collection m_items;
    };


    class Integer  : public Object
    {
    public:
        Integer () :
            Object (Type::eTypeInteger),
            m_value ()
        {
        }

        virtual ~Integer()
        {
        }

        void
        SetValue (uint64_t value)
        {
            m_value = value;
        }

        uint64_t
        GetValue ()
        {
            return m_value;
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        uint64_t m_value;
    };

    class Float  : public Object
    {
    public:
        Float () :
            Object (Type::eTypeFloat),
            m_value ()
        {
        }

        virtual ~Float()
        {
        }

        void
        SetValue (double value)
        {
            m_value = value;
        }

        double
        GetValue ()
        {
            return m_value;
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        double m_value;
    };

    class Boolean  : public Object
    {
    public:
        Boolean () :
            Object (Type::eTypeBoolean),
            m_value ()
        {
        }

        virtual ~Boolean()
        {
        }

        void
        SetValue (bool value)
        {
            m_value = value;
        }

        bool
        GetValue ()
        {
            return m_value;
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        bool m_value;
    };



    class String  : public Object
    {
    public:
        String () :
            Object (Type::eTypeString),
            m_value ()
        {
        }

        void
        SetValue (std::string string)
        {
            m_value = string;
        }

        std::string
        GetValue ()
        {
            return m_value;
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        std::string m_value;
    };

    class Dictionary : public Object
    {
    public:
        Dictionary () :
            Object (Type::eTypeDictionary),
            m_dict ()
        {
        }

        virtual ~Dictionary()
        {
        }
        size_t
        GetSize()
        {
            return m_dict.size();
        }

        ObjectSP
        GetKeys()
        {
            ObjectSP object_sp(new Array ());
            Array *array = object_sp->GetAsArray();
            collection::const_iterator iter;
            for (iter = m_dict.begin(); iter != m_dict.end(); ++iter)
            {
                ObjectSP key_object_sp(new String());
                key_object_sp->GetAsString()->SetValue(iter->first.AsCString());
                array->Push(key_object_sp);
            }
            return object_sp;
        }

        ObjectSP
        GetValueForKey (const char *key)
        {
            ObjectSP value_sp;
            if (key)
            {
                ConstString key_cs(key);
                for (collection::const_iterator iter = m_dict.begin(); iter != m_dict.end(); ++iter)
                {
                    if (key_cs == iter->first)
                    {
                        value_sp = iter->second;
                        break;
                    }
                }
            }
            return value_sp;
        }

        bool
        HasKey (const char *key)
        {
            ConstString key_cs (key);
            collection::const_iterator search = m_dict.find(key_cs);
            if (search != m_dict.end())
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        void
        AddItem (const char *key, ObjectSP value)
        {
            ConstString key_cs(key);
            m_dict[key_cs] = value;
        }

        void
        AddIntegerItem (const char *key, uint64_t value)
        {
            ObjectSP val_obj (new Integer());
            val_obj->GetAsInteger()->SetValue (value);
            AddItem (key, val_obj);
        }

        void
        AddFloatItem (const char *key, double value)
        {
            ObjectSP val_obj (new Float());
            val_obj->GetAsFloat()->SetValue (value);
            AddItem (key, val_obj);
        }

        void
        AddStringItem (const char *key, std::string value)
        {
            ObjectSP val_obj (new String());
            val_obj->GetAsString()->SetValue (value);
            AddItem (key, val_obj);
        }

        void
        AddBooleanItem (const char *key, bool value)
        {
            ObjectSP val_obj (new Boolean());
            val_obj->GetAsBoolean()->SetValue (value);
            AddItem (key, val_obj);
        }

        virtual void
        Dump (Stream &s) const;

    protected:
        typedef std::map<ConstString, ObjectSP> collection;
        collection m_dict;
    };

    class Null : public Object
    {
    public:
        Null () :
            Object (Type::eTypeNull)
        {
        }

        virtual ~Null()
        {
        }

        virtual void
        Dump (Stream &s) const;

    protected:
    };


    static ObjectSP
    ParseJSON (std::string json_text);

};  // class StructuredData


} // namespace lldb_private

#endif  // liblldb_StructuredData_h_

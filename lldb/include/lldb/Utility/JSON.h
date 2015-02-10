//===---------------------JSON.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_JSON_h_
#define utility_JSON_h_

#include "lldb/Core/Stream.h"

#include <inttypes.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "llvm/Support/Casting.h"

namespace lldb_private {
    class JSONValue
    {
    public:
        virtual void
        Write (Stream& s) = 0;
        
        typedef std::shared_ptr<JSONValue> SP;
        
        enum class Kind
        {
            String,
            Number,
            True,
            False,
            Null,
            Object,
            Array
        };
        
        JSONValue (Kind k) :
        m_kind(k)
        {}
        
        Kind
        GetKind() const
        {
            return m_kind;
        }
        
        virtual
        ~JSONValue () = default;
        
    private:
        const Kind m_kind;
    };
    
    class JSONString : public JSONValue
    {
    public:
        JSONString ();
        JSONString (const char* s);
        JSONString (const std::string& s);

        JSONString (const JSONString& s) = delete;
        JSONString&
        operator = (const JSONString& s) = delete;
        
        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONString> SP;
        
        std::string
        GetData () { return m_data; }
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::String;
        }
        
        virtual
        ~JSONString () = default;
        
    private:
        
        static std::string
        json_string_quote_metachars (const std::string&);
        
        std::string m_data;
    };

    class JSONNumber : public JSONValue
    {
    public:
        JSONNumber ();
        JSONNumber (int64_t i);
        
        JSONNumber (const JSONNumber& s) = delete;
        JSONNumber&
        operator = (const JSONNumber& s) = delete;

        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONNumber> SP;
        
        int64_t
        GetData () { return m_data; }
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::Number;
        }
        
        virtual
        ~JSONNumber () = default;
        
    private:
        int64_t m_data;
    };

    class JSONTrue : public JSONValue
    {
    public:
        JSONTrue ();

        JSONTrue (const JSONTrue& s) = delete;
        JSONTrue&
        operator = (const JSONTrue& s) = delete;
        
        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONTrue> SP;
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::True;
        }
        
        virtual
        ~JSONTrue () = default;
    };

    class JSONFalse : public JSONValue
    {
    public:
        JSONFalse ();

        JSONFalse (const JSONFalse& s) = delete;
        JSONFalse&
        operator = (const JSONFalse& s) = delete;
        
        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONFalse> SP;
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::False;
        }
        
        virtual
        ~JSONFalse () = default;
    };

    class JSONNull : public JSONValue
    {
    public:
        JSONNull ();

        JSONNull (const JSONNull& s) = delete;
        JSONNull&
        operator = (const JSONNull& s) = delete;
        
        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONNull> SP;
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::Null;
        }
        
        virtual
        ~JSONNull () = default;
    };

    class JSONObject : public JSONValue
    {
    public:
        JSONObject ();
        
        JSONObject (const JSONObject& s) = delete;
        JSONObject&
        operator = (const JSONObject& s) = delete;

        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONObject> SP;
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::Object;
        }
        
        bool
        SetObject (const std::string& key,
                   JSONValue::SP value);
        
        JSONValue::SP
        GetObject (const std::string& key);
        
        virtual
        ~JSONObject () = default;
        
    private:
        typedef std::map<std::string, JSONValue::SP> Map;
        typedef Map::iterator Iterator;
        Map m_elements;
    };

    class JSONArray : public JSONValue
    {
    public:
        JSONArray ();
        
        JSONArray (const JSONArray& s) = delete;
        JSONArray&
        operator = (const JSONArray& s) = delete;
        
        virtual void
        Write (Stream& s);
        
        typedef std::shared_ptr<JSONArray> SP;
        
        static bool classof(const JSONValue *V)
        {
            return V->GetKind() == JSONValue::Kind::Array;
        }
        
    private:
        typedef std::vector<JSONValue::SP> Vector;
        typedef Vector::iterator Iterator;
        typedef Vector::size_type Index;
        typedef Vector::size_type Size;
        
    public:
        bool
        SetObject (Index i,
                   JSONValue::SP value);
        
        bool
        AppendObject (JSONValue::SP value);
        
        JSONValue::SP
        GetObject (Index i);
        
        Size
        GetNumElements ();

        virtual
        ~JSONArray () = default;
        
        Vector m_elements;
    };
}

#endif // utility_ProcessStructReader_h_

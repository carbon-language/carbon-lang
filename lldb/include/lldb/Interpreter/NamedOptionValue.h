//===-- NamedOptionValue.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NamedOptionValue_h_
#define liblldb_NamedOptionValue_h_

// C Includes
// C++ Includes
#include <vector>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private {


    //---------------------------------------------------------------------
    // OptionValue
    //---------------------------------------------------------------------
    class OptionValue
    {
    public:
        typedef enum {
            eTypeInvalid = 0,
            eTypeArray,
            eTypeBoolean,
            eTypeDictionary,
            eTypeEnum,
            eTypeFileSpec,
            eTypeSInt64,
            eTypeUInt64,
            eTypeString
        } Type;
        
        virtual ~OptionValue ()
        {
        }
        //-----------------------------------------------------------------
        // Subclasses should override these functions
        //-----------------------------------------------------------------
        virtual Type
        GetType () = 0;
        
        virtual void
        DumpValue (Stream &strm) = 0;
        
        virtual bool
        SetValueFromCString (const char *value) = 0;
        
        virtual bool
        ResetValueToDefault () = 0;

        //-----------------------------------------------------------------
        // Subclasses should NOT override these functions as they use the
        // above functions to implement functionality
        //-----------------------------------------------------------------
        uint32_t
        GetTypeAsMask ()
        {
            return 1u << GetType();
        }
    };
    
    

    //---------------------------------------------------------------------
    // OptionValueBoolean
    //---------------------------------------------------------------------
    class OptionValueBoolean : public OptionValue
    {
        OptionValueBoolean (bool current_value, 
                            bool default_value) :
            m_current_value (current_value),
            m_default_value (default_value)
        {
        }
        
        virtual 
        ~OptionValueBoolean()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeBoolean;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_current_value = m_default_value;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        bool
        GetCurrentValue() const
        {
            return m_current_value;
        }
        
        bool
        GetDefaultValue() const
        {
            return m_default_value;
        }
        
        void
        SetCurrentValue (bool value)
        {
            m_current_value = value;
        }
        
        void
        SetDefaultValue (bool value)
        {
            m_default_value = value;
        }
        
    protected:
        bool m_current_value;
        bool m_default_value;
    };
    
    //---------------------------------------------------------------------
    // OptionValueSInt64
    //---------------------------------------------------------------------
    class OptionValueSInt64 : public OptionValue
    {
        OptionValueSInt64 (int64_t current_value, 
                           int64_t default_value) :
        m_current_value (current_value),
        m_default_value (default_value)
        {
        }
        
        virtual 
        ~OptionValueSInt64()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeSInt64;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_current_value = m_default_value;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        int64_t
        GetCurrentValue() const
        {
            return m_current_value;
        }
        
        int64_t
        GetDefaultValue() const
        {
            return m_default_value;
        }
        
        void
        SetCurrentValue (int64_t value)
        {
            m_current_value = value;
        }
        
        void
        SetDefaultValue (int64_t value)
        {
            m_default_value = value;
        }
        
    protected:
        int64_t m_current_value;
        int64_t m_default_value;
    };
    
    //---------------------------------------------------------------------
    // OptionValueUInt64
    //---------------------------------------------------------------------
    class OptionValueUInt64 : public OptionValue
    {
        OptionValueUInt64 (uint64_t current_value, 
                           uint64_t default_value) :
        m_current_value (current_value),
        m_default_value (default_value)
        {
        }
        
        virtual 
        ~OptionValueUInt64()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeUInt64;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_current_value = m_default_value;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        uint64_t
        GetCurrentValue() const
        {
            return m_current_value;
        }
        
        uint64_t
        GetDefaultValue() const
        {
            return m_default_value;
        }
        
        void
        SetCurrentValue (uint64_t value)
        {
            m_current_value = value;
        }
        
        void
        SetDefaultValue (uint64_t value)
        {
            m_default_value = value;
        }
        
    protected:
        uint64_t m_current_value;
        uint64_t m_default_value;
    };

    //---------------------------------------------------------------------
    // OptionValueString
    //---------------------------------------------------------------------
    class OptionValueString : public OptionValue
    {
        OptionValueString (const char *current_value, 
                           const char *default_value) :
            m_current_value (),
            m_default_value ()
        {
            if  (current_value && current_value[0])
                m_current_value.assign (current_value);
            if  (default_value && default_value[0])
                m_default_value.assign (default_value);
        }
        
        virtual 
        ~OptionValueString()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeString;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);

        virtual bool
        ResetValueToDefault ()
        {
            m_current_value = m_default_value;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        const char *
        GetCurrentValue() const
        {
            return m_current_value.c_str();
        }
        
        const char *
        GetDefaultValue() const
        {
            return m_default_value.c_str();
        }
        
        void
        SetCurrentValue (const char *value)
        {
            if (value && value[0])
                m_current_value.assign (value);
            else
                m_current_value.clear();
        }
        
        void
        SetDefaultValue (const char *value)
        {
            if (value && value[0])
                m_default_value.assign (value);
            else
                m_default_value.clear();
        }
        
    protected:
        std::string m_current_value;
        std::string m_default_value;
    };

    //---------------------------------------------------------------------
    // OptionValueFileSpec
    //---------------------------------------------------------------------
    class OptionValueFileSpec : public OptionValue
    {
        OptionValueFileSpec (const FileSpec &current_value, 
                             const FileSpec &default_value) :
            m_current_value (current_value),
            m_default_value (default_value)
        {
        }
        
        virtual 
        ~OptionValueFileSpec()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeFileSpec;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_current_value = m_default_value;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        const FileSpec &
        GetCurrentValue() const
        {
            return m_current_value;
        }
        
        const FileSpec &
        GetDefaultValue() const
        {
            return m_default_value;
        }
        
        void
        SetCurrentValue (const FileSpec &value)
        {
            m_current_value = value;
        }
        
        void
        SetDefaultValue (const FileSpec &value)
        {
            m_default_value = value;
        }
        
    protected:
        FileSpec m_current_value;
        FileSpec m_default_value;
    };
    
    //---------------------------------------------------------------------
    // OptionValueArray
    //---------------------------------------------------------------------
    class OptionValueArray : public OptionValue
    {
        OptionValueArray (uint32_t type_mask = UINT32_MAX) :
            m_type_mask (type_mask),
            m_values ()
        {
        }
        
        virtual 
        ~OptionValueArray()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeArray;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_values.clear();
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------

        uint32_t
        GetNumValues() const
        {
            return m_values.size();
        }

        lldb::OptionValueSP
        GetValueAtIndex (uint32_t idx) const
        {
            lldb::OptionValueSP value_sp;
            if (idx < m_values.size())
                value_sp = m_values[idx];
            return value_sp;
        }
        
        bool
        AppendValue (const lldb::OptionValueSP &value_sp)
        {
            // Make sure the value_sp object is allowed to contain
            // values of the type passed in...
            if (value_sp && (m_type_mask & value_sp->GetTypeAsMask()))
            {
                m_values.push_back(value_sp);
                return true;
            }
            return false;
        }
        
        bool
        InsertValue (uint32_t idx, const lldb::OptionValueSP &value_sp)
        {
            // Make sure the value_sp object is allowed to contain
            // values of the type passed in...
            if (value_sp && (m_type_mask & value_sp->GetTypeAsMask()))
            {
                if (idx < m_values.size())
                    m_values.insert(m_values.begin() + idx, value_sp);
                else
                    m_values.push_back(value_sp);
                return true;
            }
            return false;
        }

        bool
        ReplaceValue (uint32_t idx, const lldb::OptionValueSP &value_sp)
        {
            // Make sure the value_sp object is allowed to contain
            // values of the type passed in...
            if (value_sp && (m_type_mask & value_sp->GetTypeAsMask()))
            {
                if (idx < m_values.size())
                {
                    m_values[idx] = value_sp;
                    return true;
                }
            }
            return false;
        }

        bool
        DeleteValue (uint32_t idx)
        {
            if (idx < m_values.size())
            {
                m_values.erase (m_values.begin() + idx);
                return true;
            }
            return false;
        }
        
    protected:
        typedef std::vector<lldb::OptionValueSP> collection;
                
        uint32_t m_type_mask;
        collection m_values;
    };

    
    
    //---------------------------------------------------------------------
    // OptionValueDictionary
    //---------------------------------------------------------------------
    class OptionValueDictionary : public OptionValue
    {
        OptionValueDictionary (uint32_t type_mask = UINT32_MAX) :
            m_type_mask (type_mask),
            m_values ()
        {
        }
        
        virtual 
        ~OptionValueDictionary()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeDictionary;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual bool
        SetValueFromCString (const char *value);
        
        virtual bool
        ResetValueToDefault ()
        {
            m_values.clear();
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        uint32_t
        GetNumValues() const
        {
            return m_values.size();
        }
        
        lldb::OptionValueSP
        GetValueForKey (const ConstString &key) const
        {
            lldb::OptionValueSP value_sp;
            collection::const_iterator pos = m_values.find (key);
            if (pos != m_values.end())
                value_sp = pos->second;
            return value_sp;
        }
        
        bool
        SetValueForKey (const ConstString &key, 
                        const lldb::OptionValueSP &value_sp, 
                        bool can_replace)
        {
            // Make sure the value_sp object is allowed to contain
            // values of the type passed in...
            if (value_sp && (m_type_mask & value_sp->GetTypeAsMask()))
            {
                if (!can_replace)
                {
                    collection::const_iterator pos = m_values.find (key);
                    if (pos != m_values.end())
                        return false;
                }
                m_values[key] = value_sp;
                return true;
            }
            return false;
        }
        
        bool
        DeleteValueForKey (const ConstString &key)
        {
            collection::iterator pos = m_values.find (key);
            if (pos != m_values.end())
            {
                m_values.erase(pos);
                return true;
            }
            return false;
        }
        
    protected:
        typedef std::map<ConstString, lldb::OptionValueSP> collection;
        uint32_t m_type_mask;
        collection m_values;
    };
    


    //---------------------------------------------------------------------
    // NamedOptionValue
    //---------------------------------------------------------------------
    class NamedOptionValue
    {
    public:
        
        NamedOptionValue (NamedOptionValue *parent, const ConstString &name) :
            m_parent (parent),
            m_name (name),
            m_user_data (0)
        {
        }

        virtual
        ~NamedOptionValue ()
        {
        }
        
        NamedOptionValue *
        GetParent ()
        {
            return m_parent;
        }

        const NamedOptionValue *
        GetParent () const
        {
            return m_parent;
        }

        const ConstString &
        GetName () const
        {
            return m_name;
        }
        
        uint32_t
        GetUserData () const
        {
            return m_user_data;
        }

        void
        SetUserData (uint32_t user_data)
        {
            m_user_data = user_data;
        }

        void
        GetQualifiedName (Stream &strm);

        lldb::OptionValueSP
        GetValue ()
        {
            return m_value_sp;
        }
        
        void
        SetValue (const lldb::OptionValueSP &value_sp)
        {
            m_value_sp = value_sp;
        }
        
        OptionValue::Type
        GetValueType ();
        
        bool
        DumpValue (Stream &strm);
        
        bool
        SetValueFromCString (const char *value);
        
        bool
        ResetValueToDefault ();
        
        OptionValueBoolean *
        GetBooleanValue ();

        OptionValueSInt64 *
        GetSInt64Value ();
        
        OptionValueUInt64 *
        GetUInt64Value ();        

        OptionValueString *
        GetStringValue ();

        OptionValueFileSpec *
        GetFileSpecValue() ;

        OptionValueArray *
        GetArrayValue() ;

        OptionValueDictionary *
        GetDictionaryValue() ;

    protected:
        NamedOptionValue *m_parent;      // NULL if this is a root object
        ConstString m_name;         // Name for this setting
        uint32_t m_user_data;       // User data that can be used for anything.
        lldb::OptionValueSP m_value_sp;   // Abstract option value
    };

    
} // namespace lldb_private

#endif  // liblldb_NamedOptionValue_h_

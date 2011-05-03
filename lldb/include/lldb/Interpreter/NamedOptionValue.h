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
#include "lldb/Core/UUID.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private {

    class OptionValueBoolean;
    class OptionValueSInt64;
    class OptionValueUInt64;
    class OptionValueString;
    class OptionValueFileSpec;
    class OptionValueFormat;
    class OptionValueUUID;
    class OptionValueArray;
    class OptionValueDictionary;

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
            eTypeFormat,
            eTypeSInt64,
            eTypeUInt64,
            eTypeUUID,
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
        
        virtual Error
        SetValueFromCString (const char *value) = 0;
        
        virtual bool
        Clear () = 0;

        //-----------------------------------------------------------------
        // Subclasses should NOT override these functions as they use the
        // above functions to implement functionality
        //-----------------------------------------------------------------
        uint32_t
        GetTypeAsMask ()
        {
            return 1u << GetType();
        }
        
        static uint32_t
        ConvertTypeToMask (OptionValue::Type type)
        {
            return 1u << type;
        }

        // Get this value as a uint64_t value if it is encoded as a boolean,
        // uint64_t or int64_t. Other types will cause "fail_value" to be 
        // returned
        uint64_t
        GetUInt64Value (uint64_t fail_value, bool *success_ptr);

        OptionValueBoolean *
        GetAsBoolean ();
        
        OptionValueSInt64 *
        GetAsSInt64 ();
        
        OptionValueUInt64 *
        GetAsUInt64 ();        
        
        OptionValueString *
        GetAsString ();
        
        OptionValueFileSpec *
        GetAsFileSpec ();
        
        OptionValueFormat *
        GetAsFormat ();
        
        OptionValueUUID *
        GetAsUUID ();
        
        OptionValueArray *
        GetAsArray ();
        
        OptionValueDictionary *
        GetAsDictionary ();

        const char *
        GetStringValue (const char *fail_value = NULL);

        uint64_t
        GetUInt64Value (uint64_t fail_value = 0);
                
        lldb::Format
        GetFormatValue (lldb::Format fail_value = lldb::eFormatDefault);

        bool
        OptionWasSet () const
        {
            return m_value_was_set;
        }

    protected:
        bool m_value_was_set; // This can be used to see if a value has been set
                              // by a call to SetValueFromCString(). It is often
                              // handy to know if an option value was set from
                              // the command line or as a setting, versus if we
                              // just have the default value that was already
                              // populated in the option value.
        
    };
    
    

    //---------------------------------------------------------------------
    // OptionValueBoolean
    //---------------------------------------------------------------------
    class OptionValueBoolean : public OptionValue
    {
    public:
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        //------------------------------------------------------------------
        /// Convert to bool operator.
        ///
        /// This allows code to check a OptionValueBoolean in conditions.
        ///
        /// @code
        /// OptionValueBoolean bool_value(...);
        /// if (bool_value)
        /// { ...
        /// @endcode
        ///
        /// @return
        ///     /b True this object contains a valid namespace decl, \b 
        ///     false otherwise.
        //------------------------------------------------------------------
        operator bool() const
        {
            return m_current_value;
        }
        
        const bool &
        operator = (bool b)
        {
            m_current_value = b;
            return m_current_value;
        }

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
    public:
        OptionValueSInt64 () :
            m_current_value (0),
            m_default_value (0)
        {
        }
        
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        const int64_t &
        operator = (int64_t value)
        {
            m_current_value = value;
            return m_current_value;
        }

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
    public:
        OptionValueUInt64 () :
            m_current_value (0),
            m_default_value (0)
        {
        }

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
        // Decode a uint64_t from "value_cstr" return a OptionValueUInt64 object
        // inside of a lldb::OptionValueSP object if all goes well. If the 
        // string isn't a uint64_t value or any other error occurs, return an 
        // empty lldb::OptionValueSP and fill error in with the correct stuff.
        //---------------------------------------------------------------------
        static lldb::OptionValueSP
        Create (const char *value_cstr, Error &error);
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        const uint64_t &
        operator = (uint64_t value)
        {
            m_current_value = value;
            return m_current_value;
        }

        operator uint64_t () const
        {
            return m_current_value;
        }

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
    public:
        OptionValueString () :
            m_current_value (),
            m_default_value ()
        {
        }

        OptionValueString (const char *current_value, 
                           const char *default_value = NULL) :
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
        
        virtual Error
        SetValueFromCString (const char *value);

        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        const char *
        operator = (const char *value)
        {
            if (value && value[0])
                m_current_value.assign (value);
            else
                m_current_value.clear();
            return m_current_value.c_str();
        }

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
        AppendToCurrentValue (const char *value)
        {
            if (value && value[0])
                m_current_value.append (value);
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
    public:
        OptionValueFileSpec () :
            m_current_value (),
            m_default_value ()
        {
        }
        
        OptionValueFileSpec (const FileSpec &current_value) :
            m_current_value (current_value),
            m_default_value ()
        {
        }
        
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        FileSpec &
        GetCurrentValue()
        {
            return m_current_value;
        }

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
    // OptionValueFormat
    //---------------------------------------------------------------------
    class OptionValueFormat : public OptionValue
    {
    public:
        OptionValueFormat (lldb::Format current_value = lldb::eFormatDefault, 
                           lldb::Format default_value = lldb::eFormatDefault,
                           uint32_t current_byte_size = 0,
                           uint32_t default_byte_size = 0,
                           bool byte_size_prefix_ok = false) :
            m_current_value (current_value),
            m_default_value (default_value),
            m_current_byte_size (current_byte_size),
            m_default_byte_size (default_byte_size),
            m_byte_size_prefix_ok (byte_size_prefix_ok)
        {
        }
        
        virtual 
        ~OptionValueFormat()
        {
        }
        
        //---------------------------------------------------------------------
        // Virtual subclass pure virtual overrides
        //---------------------------------------------------------------------
        
        virtual OptionValue::Type
        GetType ()
        {
            return eTypeFormat;
        }
        
        virtual void
        DumpValue (Stream &strm);
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_current_value = m_default_value;
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        lldb::Format
        GetCurrentValue() const
        {
            return m_current_value;
        }
        
        lldb::Format 
        GetDefaultValue() const
        {
            return m_default_value;
        }
        
        void
        SetCurrentValue (lldb::Format value)
        {
            m_current_value = value;
        }
        
        void
        SetDefaultValue (lldb::Format value)
        {
            m_default_value = value;
        }
        
        uint32_t 
        GetCurrentByteSize () const
        {
            return m_current_byte_size;
        }

        uint32_t 
        GetDefaultByteSize () const
        {
            return m_default_byte_size;
        }
        
        void
        SetCurrentByteSize (uint32_t byte_size)
        {
            m_current_byte_size = byte_size;
        }
        
        void
        SetDefaultByteSize (uint32_t byte_size)
        {
            m_default_byte_size = byte_size;
        }

    protected:
        lldb::Format m_current_value;
        lldb::Format m_default_value;
        uint32_t m_current_byte_size;
        uint32_t m_default_byte_size;
        bool m_byte_size_prefix_ok;
    };
    
    
    
    //---------------------------------------------------------------------
    // OptionValueUUID
    //---------------------------------------------------------------------
    class OptionValueUUID : public OptionValue
    {
    public:
        OptionValueUUID () :
            m_uuid ()
        {
        }
        
        OptionValueUUID (const UUID &uuid) :
            m_uuid (uuid)
        {
        }
        
        virtual 
        ~OptionValueUUID()
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_uuid.Clear();
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        UUID &
        GetCurrentValue()
        {
            return m_uuid;
        }
        
        const UUID &
        GetCurrentValue() const
        {
            return m_uuid;
        }
        
        void
        SetCurrentValue (const UUID &value)
        {
            m_uuid = value;
        }
        
    protected:
        UUID m_uuid;
    };

    //---------------------------------------------------------------------
    // OptionValueArray
    //---------------------------------------------------------------------
    class OptionValueArray : public OptionValue
    {
    public:
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_values.clear();
            m_value_was_set = false;
            return true;
        }
        
        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------

        uint32_t
        GetSize () const
        {
            return m_values.size();
        }

        lldb::OptionValueSP
        operator[](uint32_t idx) const
        {
            lldb::OptionValueSP value_sp;
            if (idx < m_values.size())
                value_sp = m_values[idx];
            return value_sp;
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
    public:
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
        
        virtual Error
        SetValueFromCString (const char *value);
        
        virtual bool
        Clear ()
        {
            m_values.clear();
            m_value_was_set = false;
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
        GetValueForKey (const ConstString &key) const;
        
        //---------------------------------------------------------------------
        // String value getters and setters
        //---------------------------------------------------------------------
        const char *
        GetStringValueForKey (const ConstString &key);

        bool
        SetStringValueForKey (const ConstString &key, 
                              const char *value,
                              bool can_replace = true);

        
        bool
        SetValueForKey (const ConstString &key, 
                        const lldb::OptionValueSP &value_sp, 
                        bool can_replace = true);
        
        bool
        DeleteValueForKey (const ConstString &key);
        
    protected:
        typedef std::map<ConstString, lldb::OptionValueSP> collection;
        uint32_t m_type_mask;
        collection m_values;
    };
    

    
    //---------------------------------------------------------------------
    // OptionValueCollection
    //
    // The option value collection is a class that must be subclassed in
    // order to provide a collection of named OptionValue objects. The
    // collection is immutable (use OptionValueDictionary for mutable key
    // value pair collection). This allows classes to have some member
    // variables that are OptionValue subclasses, and still provide access
    // to setting and modifying these values from textual commands:
    //
    //    
    //    class Car : public OptionValueCollection 
    //    {
    //    public:
    //        
    //        Car () : OptionValueCollection (NULL, "car"),
    //             m_is_running_name ("running"),
    //             m_license_number_name ("license"),
    //             m_is_running (false, false),
    //             m_license_number ()
    //        {
    //        }
    //        
    //        
    //        bool
    //        GetIsRunning () const
    //        {
    //            return m_is_running.GetCurrentValue();
    //        }
    //        
    //        const char *
    //        GetLicense () const
    //        {
    //            return m_license_number.GetCurrentValue();
    //        }
    //        
    //        virtual uint32_t
    //        GetNumValues() const
    //        {
    //            return 2;
    //        }
    //        
    //        virtual ConstString
    //        GetKeyAtIndex (uint32_t idx) const
    //        {
    //            switch (idx)
    //            {
    //                case 0: return m_is_running_name;
    //                case 1: return m_license_number_name;
    //            }
    //            return ConstString();
    //        }
    //        
    //        virtual OptionValue*
    //        GetValueForKey (const ConstString &key)
    //        {
    //            if (key == m_is_running_name)
    //                return &m_is_running;
    //            else if (key == m_license_number_name)
    //                return &m_license_number;
    //            return NULL;
    //        }
    //        
    //    protected:
    //        ConstString m_is_running_name;
    //        ConstString m_license_number_name;
    //        OptionValueBoolean m_is_running;
    //        OptionValueString m_license_number;
    //        
    //    };
    //
    // As we can see above, this allows the Car class to have direct access
    // to its member variables settings m_is_running and m_license_number,
    // yet it allows them to also be available by name to our command
    // interpreter.
    //---------------------------------------------------------------------
    class OptionValueCollection
    {
    public:
        OptionValueCollection (OptionValueCollection *parent, const ConstString &name) :
            m_parent (parent),
            m_name (name)
        {
        }

        OptionValueCollection (OptionValueCollection *parent, const char *name) :
            m_parent (parent),
            m_name (name)
        {
        }

        virtual 
        ~OptionValueCollection()
        {
        }
        
        
        OptionValueCollection *
        GetParent ()
        {
            return m_parent;
        }
        
        const OptionValueCollection *
        GetParent () const
        {
            return m_parent;
        }
        
        const ConstString &
        GetName () const
        {
            return m_name;
        }

        void
        GetQualifiedName (Stream &strm);

        //---------------------------------------------------------------------
        // Subclass specific functions
        //---------------------------------------------------------------------
        
        virtual uint32_t
        GetNumValues() const = 0;
        
        virtual ConstString
        GetKeyAtIndex (uint32_t idx) const = 0;

        virtual OptionValue*
        GetValueForKey (const ConstString &key) = 0;

    protected:
        OptionValueCollection *m_parent;    // NULL if this is a root object
        ConstString m_name;                 // Name for this collection setting (if any)
    };
    
    

} // namespace lldb_private

#endif  // liblldb_NamedOptionValue_h_

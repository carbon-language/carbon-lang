//====-- UserSettingsController.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include <algorithm>

#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb_private;

UserSettingsController::UserSettingsController (const char *level_name, 
                                                const lldb::UserSettingsControllerSP &parent) :
    m_default_settings (),
    m_settings (),
    m_children (),
    m_pending_settings (),
    m_live_settings (),
    m_children_mutex (Mutex::eMutexTypeNormal),
    m_pending_settings_mutex (Mutex::eMutexTypeRecursive),
    m_live_settings_mutex (Mutex::eMutexTypeRecursive)
{
    m_settings.parent = parent;
    m_settings.level_name.SetCString (level_name);
}

UserSettingsController::~UserSettingsController ()
{
    m_live_settings.clear();
}

bool
UserSettingsController::SetGlobalVariable
(
    const ConstString &var_name,
    const char *index_value,
    const char *value,
    const SettingEntry &entry,
    const lldb::VarSetOperationType op,
    Error &err
)
{
    err.SetErrorString ("UserSettingsController has no global settings");
    return false;
}

bool
UserSettingsController::GetGlobalVariable 
(
    const ConstString &var_name, 
    StringList &value,
    Error &err
)
{
    return false;
}

bool
UserSettingsController::InitializeSettingsController (lldb::UserSettingsControllerSP &controller_sp,
                                                      SettingEntry *global_settings,
                                                      SettingEntry *instance_settings)
{
    const lldb::UserSettingsControllerSP &parent = controller_sp->GetParent ();
    if (parent)
        parent->RegisterChild (controller_sp);

    controller_sp->CreateSettingsVector (global_settings, true);
    controller_sp->CreateSettingsVector (instance_settings, false);

    controller_sp->InitializeGlobalVariables ();
    controller_sp->CreateDefaultInstanceSettings ();

    return true;
}

void
UserSettingsController::FinalizeSettingsController (lldb::UserSettingsControllerSP &controller_sp)
{
    const lldb::UserSettingsControllerSP &parent = controller_sp->GetParent ();
    if (parent)
        parent->RemoveChild (controller_sp);
}

void
UserSettingsController::InitializeGlobalVariables ()
{
    int num_entries;
    const char *prefix = GetLevelName().AsCString();

    num_entries = m_settings.global_settings.size();
    for (int i = 0; i < num_entries; ++i)
    {
        SettingEntry &entry = m_settings.global_settings[i];
        if (entry.default_value != NULL)
        {
            StreamString full_name;
            if (prefix[0] != '\0')
                full_name.Printf ("%s.%s", prefix, entry.var_name);
            else
                full_name.Printf ("%s", entry.var_name);
            SetVariable (full_name.GetData(), entry.default_value, lldb::eVarSetOperationAssign, false, "");
        }
        else if ((entry.var_type == lldb::eSetVarTypeEnum)
                 && (entry.enum_values != NULL))
        {
            StreamString full_name;
            if (prefix[0] != '\0')
                full_name.Printf ("%s.%s", prefix, entry.var_name);
            else
                full_name.Printf ("%s", entry.var_name);
            SetVariable (full_name.GetData(), entry.enum_values[0].string_value, lldb::eVarSetOperationAssign,
                         false, "");
        }
    }
}

const lldb::UserSettingsControllerSP &
UserSettingsController::GetParent ()
{
    return m_settings.parent;
}

void
UserSettingsController::RegisterChild (const lldb::UserSettingsControllerSP &child)
{
    Mutex::Locker locker (m_children_mutex);

    // Verify child is not already in m_children.
    size_t num_children = m_children.size();
    bool found = false;
    for (size_t i = 0; i < num_children; ++i)
      {
	if (m_children[i].get() == child.get())
    	    found = true;
      }

    // Add child to m_children.
    if (! found)
        m_children.push_back (child);
}

const ConstString &
UserSettingsController::GetLevelName ()
{
    return m_settings.level_name;
}

size_t
UserSettingsController::GetNumChildren ()
{
    return m_children.size();
}

const lldb::UserSettingsControllerSP
UserSettingsController::GetChildAtIndex (size_t index)
{
    if (index < m_children.size())
        return m_children[index];

    lldb::UserSettingsControllerSP dummy_value;

    return dummy_value;
}

const SettingEntry *
UserSettingsController::GetGlobalEntry (const ConstString &var_name)
{

    for (int i = 0; i < m_settings.global_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.global_settings[i];
        ConstString entry_name (entry.var_name);
        if (entry_name == var_name)
            return &entry;
    }

    return NULL;
}

const SettingEntry *
UserSettingsController::GetInstanceEntry (const ConstString &const_var_name)
{

    for (int i = 0; i < m_settings.instance_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.instance_settings[i];
        ConstString entry_name (entry.var_name);
        if (entry_name == const_var_name)
            return &entry;
    }

    return NULL;
}

void
UserSettingsController::BuildParentPrefix (std::string &parent_prefix)
{
    lldb::UserSettingsControllerSP parent = GetParent();
    if (parent.get() != NULL)
    {
        parent->BuildParentPrefix (parent_prefix);
        if (parent_prefix.length() > 0)
            parent_prefix.append (".");
    }
    parent_prefix.append (GetLevelName().AsCString());
}

void
UserSettingsController::RemoveChild (const lldb::UserSettingsControllerSP &child)
{
    Mutex::Locker locker (m_children_mutex);
    std::vector<lldb::UserSettingsControllerSP>::iterator pos, end = m_children.end();

   for (pos = m_children.begin(); pos != end; ++pos)
   {
      lldb::UserSettingsControllerSP entry = *pos;
      if (entry == child)
      {
          m_children.erase (pos);
          break;
      }
   }
}

Error
UserSettingsController::SetVariable (const char *full_dot_name, 
                                     const char *value, 
                                     const lldb::VarSetOperationType op,
                                     const bool override,
                                     const char *debugger_instance_name,
                                     const char *index_value)
{
    Error err;
    ConstString const_var_name;
    const ConstString &default_name = InstanceSettings::GetDefaultName();

    Args names = UserSettingsController::BreakNameIntoPieces (full_dot_name);
    int num_pieces = names.GetArgumentCount();

    if (num_pieces < 1)
    {
        err.SetErrorStringWithFormat ("'%s' is not a valid variable name; cannot assign value.\n", full_dot_name);
        return err;
    }

    ConstString prefix (names.GetArgumentAtIndex (0));

    if ((prefix == m_settings.level_name)
        || (m_settings.level_name.GetLength() == 0))
    {

        if (prefix == m_settings.level_name)
        {
            names.Shift ();
            num_pieces = names.GetArgumentCount();
        }

        if (num_pieces == 0)
        {
            err.SetErrorString ("No variable name specified; cannot assign value.\n");
            return err;
        }
        else if (num_pieces == 1)
        {

            // Must be one of the class-wide settings.

            const_var_name.SetCString (names.GetArgumentAtIndex (0));
            const SettingEntry *entry = GetGlobalEntry (const_var_name);
            if (entry)
            {
                UserSettingsController::VerifyOperationForType (entry->var_type, op, const_var_name, err);

                if (err.Fail())
                    return err;

                if ((value == NULL || value[0] == '\0')
                    && (op == lldb::eVarSetOperationAssign))
                {
                    if (entry->var_type != lldb::eSetVarTypeEnum)
                        value = entry->default_value;
                    else
                        value = entry->enum_values[0].string_value;
                }
                SetGlobalVariable (const_var_name, index_value, value, *entry, op, err);
            }
            else
            {
                // MIGHT be instance variable, to be for ALL instances.

                entry = GetInstanceEntry (const_var_name);
                if (entry == NULL)
                {
                    err.SetErrorStringWithFormat ("Unable to find variable '%s.%s'; cannot assign value.\n",
                                                  prefix.AsCString(), const_var_name.AsCString());
                    return err;
                }
                else
                {
                    UserSettingsController::VerifyOperationForType (entry->var_type, op, const_var_name, err);

                    if (err.Fail())
                        return err;

                    if ((value == NULL || value[0] == '\0')
                        && (op == lldb::eVarSetOperationAssign))
                    {
                        if (entry->var_type != lldb::eSetVarTypeEnum)
                            value = entry->default_value;
                        else
                            value = entry->enum_values[0].string_value;
                    }

                    if ((m_settings.level_name.GetLength() > 0)
                        || strlen (debugger_instance_name) == 0)
                      {
                        // Set the default settings
                        m_default_settings->UpdateInstanceSettingsVariable (const_var_name, index_value, value, 
                                                                            default_name, *entry, op, err, true);
                      }
                    else
                      {
                        // We're at the Debugger level; find the correct debugger instance and set those settings
                        StreamString tmp_name;
                        if (debugger_instance_name[0] != '[')
                            tmp_name.Printf ("[%s]", debugger_instance_name);
                        else
                            tmp_name.Printf ("%s", debugger_instance_name);
                        ConstString dbg_name (tmp_name.GetData());
                        InstanceSettings *dbg_settings = FindSettingsForInstance (dbg_name);
                        if (dbg_settings)
                            dbg_settings->UpdateInstanceSettingsVariable (const_var_name, index_value, value, dbg_name,
                                                                          *entry, op, err, false);
                      }

                    if (override)
                    {
                        OverrideAllInstances (const_var_name, value, op, index_value, err);

                        // Update all pending records as well.
//                        std::map<std::string, lldb::InstanceSettingsSP>::iterator pos, end = m_pending_settings.end();
//                        for (pos = m_pending_settings.begin(); pos != end; end++)
//                        {
//                            const ConstString instance_name (pos->first.c_str());
//                            lldb::InstanceSettingsSP setting_sp = pos->second;
//                            setting_sp->UpdateInstanceSettingsVariable (const_var_name, index_value, value, 
//                                                                        instance_name, *entry, op, err, true);
//                        }
                    }
                }
            }
        }
        else
        {
            // Either a child's setting or an instance setting.

            if (names.GetArgumentAtIndex(0)[0] == '[')
            {
                // An instance setting.  Supposedly.

                ConstString instance_name (names.GetArgumentAtIndex (0));
                
                // First verify that there is only one more name.
                
                names.Shift();
                
                if (names.GetArgumentCount() != 1)
                {
                    err.SetErrorStringWithFormat ("Invalid variable name format '%s'; cannot assign value.\n",
                                                  full_dot_name);
                    return err;
                }
                
                // Next verify that it is a valid instance setting name.
                
                const_var_name.SetCString (names.GetArgumentAtIndex (0));
                const SettingEntry *entry = GetInstanceEntry (const_var_name);

                if (entry == NULL)
                {
                    err.SetErrorStringWithFormat ("Unknown instance variable '%s'; cannot assign value.\n",
                                                  const_var_name.AsCString());
                    return err;
                }

                UserSettingsController::VerifyOperationForType (entry->var_type, op, const_var_name, err);

                if (err.Fail())
                    return err;

                if ((value == NULL || value[0] == '\0')
                    && (op == lldb::eVarSetOperationAssign))
                {
                    if (entry->var_type != lldb::eSetVarTypeEnum)
                        value = entry->default_value;
                    else
                        value = entry->enum_values[0].string_value;
                }

                // Now look for existing instance with given instance name; if not found, find or create pending
                // setting for instance with given name.

                InstanceSettings *current_settings = FindSettingsForInstance (instance_name);

                if (current_settings != NULL)
                {
                    current_settings->UpdateInstanceSettingsVariable (const_var_name, index_value, value, 
                                                                      instance_name, *entry, op, err, false);

                }
                else
                {
                    // Instance does not currently exist; make or update a pending setting for it.
                    lldb::InstanceSettingsSP current_settings_sp = PendingSettingsForInstance (instance_name);

                    // Now we have a settings record, update it appropriately.

                    current_settings_sp->UpdateInstanceSettingsVariable (const_var_name, index_value, value, 
                                                                         instance_name, *entry, op, err, true);
                    
                    {   // Scope for mutex.
                        Mutex::Locker locker (m_pending_settings_mutex);
                        m_pending_settings[instance_name.AsCString()] = current_settings_sp;
                    }
 
                    if (override)
                    {
                        OverrideAllInstances (const_var_name, value, op, index_value, err);
                        
                        // Update all pending records as well.
                        std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;
                        std::map<std::string, lldb::InstanceSettingsSP>::iterator end = m_pending_settings.end();
                        for (pos = m_pending_settings.begin(); pos != end; end++)
                        {
                            const ConstString tmp_inst_name (pos->first.c_str());
                            lldb::InstanceSettingsSP setting_sp = pos->second;
                            setting_sp->UpdateInstanceSettingsVariable (const_var_name, index_value, value, 
                                                                        tmp_inst_name, *entry, op, err, true);
                        }
                    }
                }
            }
            else
            {
                // A child setting.
                lldb::UserSettingsControllerSP child;
                ConstString child_prefix (names.GetArgumentAtIndex (0));
                int num_children = GetNumChildren();
                bool found = false;
                for (int i = 0; i < num_children && !found; ++i)
                {
                    child = GetChildAtIndex (i);
                    ConstString current_prefix = child->GetLevelName();
                    if (current_prefix == child_prefix)
                    {
                        found = true;
                        std::string new_name;
                        for (int j = 0; j < names.GetArgumentCount(); ++j)
                        {
                            if (j > 0)
                                new_name += '.';
                            new_name += names.GetArgumentAtIndex (j);
                        }
                        return child->SetVariable (new_name.c_str(), value, op, override, debugger_instance_name,
                                                   index_value);
                    }
                }
                if (!found)
                {
                    err.SetErrorStringWithFormat ("Unable to find variable '%s'; cannot assign value.\n", 
                                                  full_dot_name);
                    return err;
                }
            }
        }
    }
    else
    {
        err.SetErrorStringWithFormat ("'%s' is not a valid level name; was expecting '%s'. Cannot assign value.\n",
                                      prefix.AsCString(), m_settings.level_name.AsCString());
    }

    return err;
}

StringList
UserSettingsController::GetVariable 
(
    const char *full_dot_name, 
    lldb::SettableVariableType &var_type, 
    const char *debugger_instance_name,
    Error &err
)
{
    Args names = UserSettingsController::BreakNameIntoPieces (full_dot_name);
    ConstString const_var_name;
    StringList value;

    int num_pieces = names.GetArgumentCount();

    ConstString prefix (names.GetArgumentAtIndex (0));
    const_var_name.SetCString (names.GetArgumentAtIndex (num_pieces - 1));

    const SettingEntry *global_entry = GetGlobalEntry (const_var_name);
    const SettingEntry *instance_entry = GetInstanceEntry (const_var_name);

    if ((prefix != m_settings.level_name)
        && (m_settings.level_name.GetLength () > 0))
    {
        err.SetErrorString ("Invalid variable name");
        return value;
    }

    // prefix name matched; remove it from names.
    if (m_settings.level_name.GetLength() > 0)
        names.Shift();

    // Should we pass this off to a child?  If there is more than one name piece left, and the next name piece
    // matches a child prefix, then yes.

    lldb::UserSettingsControllerSP child;
    if (names.GetArgumentCount() > 1)
    {
        ConstString child_prefix (names.GetArgumentAtIndex (0));
        bool found = false;
        for (int i = 0; i < m_children.size() && !found; ++i)
        {
            if (child_prefix == m_children[i]->GetLevelName())
            {
                found = true;
                child = m_children[i];
                std::string new_name;
                for (int j = 0; j < names.GetArgumentCount(); ++j)
                {
                    if (j > 0)
                        new_name += '.';
                    new_name += names.GetArgumentAtIndex (j);
                }
                return child->GetVariable (new_name.c_str(), var_type, debugger_instance_name, err);
            }
        }

        if (!found)
        {
            // Cannot be handled by a child, because name did not match any child prefixes.
            // Cannot be a class-wide variable because there are too many name pieces.

            if (instance_entry != NULL)
            {
                var_type = instance_entry->var_type;
                ConstString instance_name (names.GetArgumentAtIndex (0));
                InstanceSettings *current_settings = FindSettingsForInstance (instance_name);

                if (current_settings != NULL)
                {
                    current_settings->GetInstanceSettingsValue (*instance_entry, const_var_name, value, &err);
                }
                else
                {
                    // Look for instance name setting in pending settings.

                    std::string inst_name_str = instance_name.AsCString();
                    std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;

                    pos = m_pending_settings.find (inst_name_str);
                    if (pos != m_pending_settings.end())
                    {
                        lldb::InstanceSettingsSP settings_sp = pos->second;
                        settings_sp->GetInstanceSettingsValue (*instance_entry, const_var_name,  value, &err);
                    }
                    else 
                    {
                        if (m_settings.level_name.GetLength() > 0)
                        {
                            // No valid instance name; assume they want the default settings.
                            m_default_settings->GetInstanceSettingsValue (*instance_entry, const_var_name, value, &err);
                        }
                        else
                        {
                            // We're at the Debugger level;  use the debugger's instance settings.
                            StreamString tmp_name;
                            if (debugger_instance_name[0] != '[')
                                tmp_name.Printf ("[%s]", debugger_instance_name);
                            else
                                tmp_name.Printf ("%s", debugger_instance_name);
                            ConstString dbg_name (debugger_instance_name);
                            InstanceSettings *dbg_settings = FindSettingsForInstance (dbg_name);
                            if (dbg_settings)
                                dbg_settings->GetInstanceSettingsValue (*instance_entry, const_var_name, value, &err);
                        }
                    }
                }
            }
            else
                err.SetErrorString ("Invalid variable name");
        }
    }
    else
    {
        // Only one name left.  It must belong to the current level, or be an error.
        if ((global_entry == NULL)
            && (instance_entry == NULL))
        {
            err.SetErrorString ("Invalid variable name");
        }
        else if (global_entry)
        {
            var_type = global_entry->var_type;
            GetGlobalVariable (const_var_name, value, err);
        }
        else if (instance_entry)
        {
            var_type = instance_entry->var_type;
            if (m_settings.level_name.GetLength() > 0)
                m_default_settings->GetInstanceSettingsValue  (*instance_entry, const_var_name, value, &err);
            else
            {
                // We're at the Debugger level;  use the debugger's instance settings.
                StreamString tmp_name;
                if (debugger_instance_name[0] != '[')
                    tmp_name.Printf ("[%s]", debugger_instance_name);
                else
                    tmp_name.Printf ("%s", debugger_instance_name);
                ConstString dbg_name (tmp_name.GetData());
                InstanceSettings *dbg_settings = FindSettingsForInstance (dbg_name);
                if (dbg_settings)
                    dbg_settings->GetInstanceSettingsValue (*instance_entry, const_var_name, value, &err);
            }
        }
    }

    return value;
}

void
UserSettingsController::RemovePendingSettings (const ConstString &instance_name)
{
    StreamString tmp_name;

    // Add surrounding brackets to instance name if not already present.

    if (instance_name.AsCString()[0] != '[')
        tmp_name.Printf ("[%s]", instance_name.AsCString());
    else
        tmp_name.Printf ("%s", instance_name.AsCString());

    std::string instance_name_str (tmp_name.GetData());
    std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;
    Mutex::Locker locker (m_pending_settings_mutex);

    m_pending_settings.erase (instance_name_str);
}

const lldb::InstanceSettingsSP &
UserSettingsController::FindPendingSettings (const ConstString &instance_name)
{
    std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;
    StreamString tmp_name;

    // Add surrounding brackets to instance name if not already present.

    if (instance_name.AsCString()[0] != '[')
        tmp_name.Printf ("[%s]", instance_name.AsCString());
    else
        tmp_name.Printf ("%s", instance_name.AsCString());

    std::string instance_name_str (tmp_name.GetData());  // Need std::string for std::map look-up

    {   // Scope for mutex.
        Mutex::Locker locker (m_pending_settings_mutex);

        pos = m_pending_settings.find (instance_name_str);
        if (pos != m_pending_settings.end())
            return pos->second;
    }

    return m_default_settings;
}

void
UserSettingsController::CreateDefaultInstanceSettings ()
{
    Error err;
    const ConstString &default_name = InstanceSettings::GetDefaultName();
    for (int i = 0; i < m_settings.instance_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.instance_settings[i];
        ConstString var_name (entry.var_name);
        const char *value = entry.default_value;

        if (entry.var_type == lldb::eSetVarTypeEnum)
            value = entry.enum_values[0].string_value;
  
        m_default_settings->UpdateInstanceSettingsVariable (var_name, NULL, value, default_name, entry, 
                                                            lldb::eVarSetOperationAssign, err, true);
    } 
}

void
UserSettingsController::CopyDefaultSettings (const lldb::InstanceSettingsSP &actual_settings,
                                             const ConstString &instance_name,
                                             bool pending)
{
    Error err;
    for (int i = 0; i < m_settings.instance_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.instance_settings[i];
        ConstString var_name (entry.var_name);
        StringList value;
        m_default_settings->GetInstanceSettingsValue (entry, var_name, value, NULL);

        std::string value_str;
        if (value.GetSize() == 1)
            value_str.append (value.GetStringAtIndex (0));
        else if (value.GetSize() > 1)
        {
            for (int j = 0; j < value.GetSize(); ++j)
            {
                if (j > 0)
                    value_str.append (" ");
              value_str.append (value.GetStringAtIndex (j));
            }
        }

        actual_settings->UpdateInstanceSettingsVariable (var_name, NULL, value_str.c_str(), instance_name, entry, 
                                                         lldb::eVarSetOperationAssign, err, pending);

    }
}

lldb::InstanceSettingsSP
UserSettingsController::PendingSettingsForInstance (const ConstString &instance_name)
{
    std::string name_str (instance_name.AsCString());
    std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;
    Mutex::Locker locker (m_pending_settings_mutex);

    pos = m_pending_settings.find (name_str);
    if (pos != m_pending_settings.end())
    {
        lldb::InstanceSettingsSP settings_sp = pos->second;
        return settings_sp;
    }
    else
    {
        lldb::InstanceSettingsSP new_settings_sp = CreateInstanceSettings (instance_name.AsCString());
        CopyDefaultSettings (new_settings_sp, instance_name, true);
        m_pending_settings[name_str] = new_settings_sp;
        return new_settings_sp;
    }
    
    // Should never reach this line.

    lldb::InstanceSettingsSP dummy;

    return dummy;
}

void
UserSettingsController::GetAllDefaultSettingValues (StreamString &result_stream)
{
    std::string parent_prefix;
    BuildParentPrefix (parent_prefix);
    const char *prefix = parent_prefix.c_str();

    for (int i = 0; i < m_settings.instance_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.instance_settings[i];
        ConstString var_name (entry.var_name);
        StringList tmp_value;
        m_default_settings->GetInstanceSettingsValue (entry, var_name, tmp_value, NULL);

        StreamString value_string;
        bool multi_value = false;

        if (tmp_value.GetSize() == 1)
            value_string.Printf ("%s", tmp_value.GetStringAtIndex (0));
        else
        {
            for (int j = 0; j < tmp_value.GetSize(); ++j)
            {
                if (entry.var_type == lldb::eSetVarTypeArray)
                    value_string.Printf ("\n  [%d]: '%s'", j, tmp_value.GetStringAtIndex (j));
                else if (entry.var_type == lldb::eSetVarTypeDictionary)
                    value_string.Printf ("\n  '%s'", tmp_value.GetStringAtIndex (j));
            }
            multi_value = true;
        }

        if (! parent_prefix.empty())
        {
            if (multi_value)
                result_stream.Printf ("%s.%s (%s):%s\n", prefix, var_name.AsCString(),
                                      UserSettingsController::GetTypeString (entry.var_type), value_string.GetData());
            else
                result_stream.Printf ("%s.%s (%s) = '%s'\n", prefix, var_name.AsCString(),
                                      UserSettingsController::GetTypeString (entry.var_type), value_string.GetData());
        }
    }
}

void
UserSettingsController::GetAllPendingSettingValues (StreamString &result_stream)
{
    std::map<std::string, lldb::InstanceSettingsSP>::iterator pos;

    std::string parent_prefix;
    BuildParentPrefix (parent_prefix);
    const char *prefix = parent_prefix.c_str();
    
    for (pos = m_pending_settings.begin(); pos != m_pending_settings.end(); ++pos)
    {
        std::string tmp_name = pos->first;
        lldb::InstanceSettingsSP settings_sp = pos->second;

        const ConstString instance_name (tmp_name.c_str());

        for (int i = 0; i < m_settings.instance_settings.size(); ++i)
        {
            SettingEntry &entry = m_settings.instance_settings[i];
            ConstString var_name (entry.var_name);
            StringList tmp_value;
            settings_sp->GetInstanceSettingsValue (entry, var_name, tmp_value, NULL);

            StreamString value_str;

            if (tmp_value.GetSize() == 0)
                value_str.Printf ("");
            else if (tmp_value.GetSize() == 1)
                value_str.Printf ("%s", tmp_value.GetStringAtIndex (0));
            else
            {
                for (int j = 0; j < tmp_value.GetSize(); ++j)
                    value_str.Printf  ("%s ", tmp_value.GetStringAtIndex (j));
            }
            
            if (parent_prefix.length() > 0)
            {
                result_stream.Printf ("%s.%s.%s (%s) = '%s' [pending]\n", prefix, instance_name.AsCString(), 
                                      var_name.AsCString(), UserSettingsController::GetTypeString (entry.var_type),
                                      value_str.GetData());
            }
            else
            {
                result_stream.Printf ("%s (%s) = '%s' [pending]\n", var_name.AsCString(),
                                      UserSettingsController::GetTypeString (entry.var_type), 
                                      value_str.GetData());                                      
            }
        }
    }
}

InstanceSettings *
UserSettingsController::FindSettingsForInstance (const ConstString &instance_name)
{
    std::string instance_name_str (instance_name.AsCString());
    std::map<std::string, InstanceSettings *>::iterator pos;

    pos = m_live_settings.find (instance_name_str);
    if (pos != m_live_settings.end ())
      {
        InstanceSettings *settings = pos->second;
        return settings;
      }

    return NULL;
}

void
UserSettingsController::GetAllInstanceVariableValues (CommandInterpreter &interpreter,
                                                      StreamString &result_stream)
{
    std::map<std::string, InstanceSettings *>::iterator pos;
    std::string parent_prefix;
    BuildParentPrefix (parent_prefix);
    const char *prefix = parent_prefix.c_str();
    StreamString description;

    for (pos = m_live_settings.begin(); pos != m_live_settings.end(); ++pos)
    {
        std::string instance_name = pos->first;
        InstanceSettings *settings = pos->second;

        for (int i = 0; i < m_settings.instance_settings.size(); ++i)
        {
            SettingEntry &entry = m_settings.instance_settings[i];
            const ConstString var_name (entry.var_name);
            StringList tmp_value;
            settings->GetInstanceSettingsValue (entry, var_name, tmp_value, NULL);
            StreamString tmp_value_str;

            if (tmp_value.GetSize() == 0)
                tmp_value_str.Printf ("");
            else if (tmp_value.GetSize() == 1)
                tmp_value_str.Printf ("%s", tmp_value.GetStringAtIndex (0));
            else
            {
                for (int j = 0; j < tmp_value.GetSize(); ++j)
                    tmp_value_str.Printf ("%s ",tmp_value.GetStringAtIndex (j));
            }
            
            description.Clear();
            if (parent_prefix.length() > 0)
            {
                description.Printf ("%s.%s.%s (%s) = '%s'", prefix, instance_name.c_str(), var_name.AsCString(), 
                                    UserSettingsController::GetTypeString (entry.var_type),
                                    tmp_value_str.GetData());
            }
            else
            {
                description.Printf ("%s (%s) = '%s'", var_name.AsCString(),
                                    UserSettingsController::GetTypeString (entry.var_type), tmp_value_str.GetData());
            }
            result_stream.Printf ("%s\n", description.GetData());
        }
    }
}

void
UserSettingsController::OverrideAllInstances (const ConstString &var_name,
                                              const char *value,
                                              lldb::VarSetOperationType op,
                                              const char *index_value,
                                              Error &err)
{
    std::map<std::string, InstanceSettings *>::iterator pos;
    StreamString description;

    for (pos = m_live_settings.begin(); pos != m_live_settings.end(); ++pos)
    {
        InstanceSettings *settings = pos->second;
        StreamString tmp_name;
        tmp_name.Printf ("[%s]", settings->GetInstanceName().AsCString());
        const ConstString instance_name (tmp_name.GetData());
        const SettingEntry *entry = GetInstanceEntry (var_name);
        settings->UpdateInstanceSettingsVariable (var_name, index_value, value, instance_name, *entry, op, err, false);

    }
}

void
UserSettingsController::RegisterInstanceSettings (InstanceSettings *instance_settings)
{
    Mutex::Locker locker (m_live_settings_mutex);
    StreamString tmp_name;
    tmp_name.Printf ("[%s]", instance_settings->GetInstanceName().AsCString());
    const ConstString instance_name (tmp_name.GetData());
    std::string instance_name_str (instance_name.AsCString());
    if (instance_name_str.compare (InstanceSettings::GetDefaultName().AsCString()) != 0)
        m_live_settings[instance_name_str] = instance_settings;
}

void
UserSettingsController::UnregisterInstanceSettings (InstanceSettings *instance)
{
    Mutex::Locker locker (m_live_settings_mutex);
    StreamString tmp_name;
    tmp_name.Printf ("[%s]", instance->GetInstanceName().AsCString());
    std::string instance_name (tmp_name.GetData());

    std::map <std::string, InstanceSettings *>::iterator pos;

    pos = m_live_settings.find (instance_name);
    if (pos != m_live_settings.end())
        m_live_settings.erase (pos);
}

void
UserSettingsController::CreateSettingsVector (const SettingEntry *table,
                                              bool global)
{
    int i = 0;
    while (table[i].var_name != NULL)
    {
        const SettingEntry &table_entry = table[i];
        ConstString const_var_name (table_entry.var_name);
        SettingEntry new_entry;
        
        new_entry = table_entry;
        new_entry.var_name = const_var_name.AsCString();
        
        if (global)
            m_settings.global_settings.push_back (new_entry);
        else
            m_settings.instance_settings.push_back (new_entry);

        ++i;
    }
}

//----------------------------------------------------------------------
// UserSettingsController static methods
//----------------------------------------------------------------------

int
FindMaxNameLength (std::vector<SettingEntry> table)
{
    int max_length = 1;

    for (int i = 0; i < table.size(); ++i)
    {
        int len = strlen (table[i].var_name);
        if (len > max_length)
            max_length = len;
    }

    return max_length;
}

const char *
UserSettingsController::GetTypeString (lldb::SettableVariableType var_type)
{
    switch (var_type)
    {
        case lldb::eSetVarTypeInt:
            return "int";
        case lldb::eSetVarTypeBoolean:
            return "boolean";
        case lldb::eSetVarTypeString:
            return "string";
        case lldb::eSetVarTypeArray:
            return "array";
        case lldb::eSetVarTypeDictionary:
            return "dictionary";
        case lldb::eSetVarTypeEnum:
            return "enum";
        case lldb::eSetVarTypeNone:
            return "no type";
    }

    return "";
}

void
UserSettingsController::PrintEnumValues (const lldb::OptionEnumValueElement *enum_values, Stream &str)
{
    int i = 0;
    while (enum_values[i].string_value != NULL)
    {
        str.Printf ("%s ", enum_values[i].string_value);
        ++i;
    }
  
}

void
UserSettingsController::FindAllSettingsDescriptions (CommandInterpreter &interpreter,
                                                     lldb::UserSettingsControllerSP root, 
                                                     std::string &current_prefix, 
                                                     StreamString &result_stream,
                                                     Error &err)
{
    // Write out current prefix line.
    StreamString prefix_line;
    StreamString description;
    uint32_t max_len;
    int num_entries = root->m_settings.global_settings.size();

    max_len = FindMaxNameLength (root->m_settings.global_settings);

    if (! current_prefix.empty())
        result_stream.Printf ("\n'%s' variables:\n\n", current_prefix.c_str());
    else
        result_stream.Printf ("\nTop level variables:\n\n");
        
    if (num_entries > 0)
    {
        // Write out all "global" variables.
        for (int i = 0; i < num_entries; ++i)
        {
            SettingEntry entry = root->m_settings.global_settings[i];
            description.Clear();
            if (entry.var_type == lldb::eSetVarTypeEnum)
            {
                StreamString enum_values_str;
                UserSettingsController::PrintEnumValues (entry.enum_values, enum_values_str);
                description.Printf ("[static, enum] %s.  Valid values: {%s}  (default: '%s')", entry.description, 
                                    enum_values_str.GetData(), entry.enum_values[0].string_value);
            }
            else if (entry.default_value != NULL)
                description.Printf ("[static, %s] %s (default: '%s')", GetTypeString (entry.var_type), 
                                    entry.description, entry.default_value);

            else
                description.Printf ("[static, %s] %s (default: '')", GetTypeString (entry.var_type), 
                                    entry.description);
            interpreter.OutputFormattedHelpText (result_stream, entry.var_name, "--", description.GetData(), 
                                                 max_len);
        }
    }

    num_entries = root->m_settings.instance_settings.size();
    max_len = FindMaxNameLength (root->m_settings.instance_settings);
   
    if (num_entries > 0)
    {
        // Write out all instance variables.
        for (int i = 0; i < num_entries; ++i)
        {
            SettingEntry entry = root->m_settings.instance_settings[i];
            description.Clear();
            if (entry.var_type == lldb::eSetVarTypeEnum)
            {
                StreamString enum_values_str;
                UserSettingsController::PrintEnumValues (entry.enum_values, enum_values_str);
                description.Printf ("[instance, enum] %s.  Valid values: {%s} (default: '%s')", entry.description,
                                    enum_values_str.GetData(), entry.enum_values[0].string_value);
            }
            else if (entry.default_value != NULL)
                description.Printf ("[instance, %s] %s (default: '%s')", GetTypeString (entry.var_type), 
                                    entry.description, entry.default_value);
            else
                description.Printf ("[instance, %s] %s (default: '')", GetTypeString (entry.var_type), 
                                    entry.description);
            interpreter.OutputFormattedHelpText (result_stream, entry.var_name, "--", description.GetData(), 
                                                 max_len);
        }

    }
    
    // Now, recurse across all children.
    int num_children = root->GetNumChildren();
    for (int i = 0; i < num_children; ++i)
    {
        lldb::UserSettingsControllerSP child = root->GetChildAtIndex (i);
        
        if (child)
        {
            ConstString child_prefix = child->GetLevelName();
            StreamString new_prefix;
            if (! current_prefix.empty())
                new_prefix.Printf ("%s.%s", current_prefix.c_str(), child_prefix.AsCString());
            else
                new_prefix.Printf ("%s", child_prefix.AsCString());
            std::string new_prefix_str = new_prefix.GetData();
            UserSettingsController::FindAllSettingsDescriptions (interpreter, child, new_prefix_str, result_stream, 
                                                                 err);
        }
    }
}

void
UserSettingsController::FindSettingsDescriptions (CommandInterpreter &interpreter,
                                                  lldb::UserSettingsControllerSP root,
                                                  std::string &current_prefix,
                                                  const char *search_name,
                                                  StreamString &result_stream,
                                                  Error &err)
{
    Args names = UserSettingsController::BreakNameIntoPieces (search_name);
    int num_pieces = names.GetArgumentCount ();

    if (num_pieces == 0)
        return;

    if (root->GetLevelName().GetLength() > 0)
    {
        ConstString prefix (names.GetArgumentAtIndex (0));
        if (prefix != root->GetLevelName())
        {
            std::string parent_prefix;
            root->BuildParentPrefix (parent_prefix);
            err.SetErrorStringWithFormat ("Cannot find match for '%s.%s'\n", parent_prefix.c_str(), 
                                          prefix.AsCString());
            return;
        }
        else
        {
            names.Shift();
            --num_pieces;
        }
    }

    // If there's nothing left then dump all global and instance descriptions for this root.
    if (num_pieces == 0)
    {
        StreamString prefix_line;
        StreamString description;
        uint32_t max_len;
        int num_entries = root->m_settings.global_settings.size();
        
        max_len = FindMaxNameLength (root->m_settings.global_settings);

        result_stream.Printf ("\n'%s' variables:\n\n", search_name);
        
        if (num_entries > 0)
        {
            // Write out all "global" variables.
            for (int i = 0; i < num_entries; ++i)
            {
                SettingEntry entry = root->m_settings.global_settings[i];
                description.Clear();
                if (entry.var_type == lldb::eSetVarTypeEnum)
                {
                    StreamString enum_values_str;
                    UserSettingsController::PrintEnumValues (entry.enum_values, enum_values_str);
                    description.Printf ("[static, enum] %s.  Valid values: {%s}  (default: '%s')", entry.description, 
                                        enum_values_str.GetData(), entry.enum_values[0].string_value);
                }
                else if (entry.default_value != NULL)
                    description.Printf ("[static, %s] %s (default: '%s')", GetTypeString (entry.var_type), 
                                        entry.description, entry.default_value);
                else
                    description.Printf ("[static, %s] %s (default: '')", GetTypeString (entry.var_type), 
                                        entry.description);
                interpreter.OutputFormattedHelpText (result_stream, entry.var_name, "--", description.GetData(), 
                                                     max_len);
            }
        }
        
        num_entries = root->m_settings.instance_settings.size();
        max_len = FindMaxNameLength (root->m_settings.instance_settings);
        
        if (num_entries > 0)
        {
            // Write out all instance variables.
            for (int i = 0; i < num_entries; ++i)
            {
                SettingEntry entry = root->m_settings.instance_settings[i];
                description.Clear();
                if (entry.var_type == lldb::eSetVarTypeEnum)
                {
                    StreamString enum_values_str;
                    UserSettingsController::PrintEnumValues (entry.enum_values, enum_values_str);
                    description.Printf ("[instance, enum] %s.  Valid values: {%s} (default: '%s')", entry.description,
                                        enum_values_str.GetData(), entry.enum_values[0].string_value);
                }
                else if (entry.default_value != NULL)
                    description.Printf ("[instance, %s] %s (default: '%s')", GetTypeString (entry.var_type), 
                                        entry.description, entry.default_value);
                else
                    description.Printf ("[instance, %s] %s (default: '')", GetTypeString (entry.var_type), 
                                        entry.description);
                interpreter.OutputFormattedHelpText (result_stream, entry.var_name, "--", description.GetData(), 
                                                     max_len);
            }
        }
    }
    else if (num_pieces == 1)
    {
        ConstString var_name (names.GetArgumentAtIndex (0));
        bool is_global = false;

        const SettingEntry *setting_entry = root->GetGlobalEntry (var_name);

        if (setting_entry == NULL)
            setting_entry = root->GetInstanceEntry (var_name);
        else
            is_global = true;

        // Check to see if it is a global or instance variable name.
        if (setting_entry != NULL)
        {
            StreamString description;
            if (setting_entry->var_type == lldb::eSetVarTypeEnum)
            {
                StreamString enum_values_str;
                UserSettingsController::PrintEnumValues (setting_entry->enum_values, enum_values_str);
                description.Printf ("[%s, enum] %s.  Valid values: {%s}  (default: '%s')", 
                                    (is_global ? "static" : "instance"), 
                                    setting_entry->description, 
                                    enum_values_str.GetData(), setting_entry->enum_values[0].string_value);
            }
            else if (setting_entry->default_value != NULL)
                description.Printf ("[%s, %s] %s (default: '%s')", 
                                    (is_global ? "static" : "instance"), 
                                    GetTypeString (setting_entry->var_type), 
                                    setting_entry->description, setting_entry->default_value);
            else
                description.Printf ("[%s, %s] %s (default: '')", 
                                    (is_global ? "static" : "instance"), 
                                    GetTypeString (setting_entry->var_type), 
                                    setting_entry->description);
            interpreter.OutputFormattedHelpText (result_stream, setting_entry->var_name, "--", description.GetData(), 
                                                 var_name.GetLength());
        }
        else
        {
            // It must be a child name.
            int num_children = root->GetNumChildren();
            bool found = false;
            for (int i = 0; i < num_children && !found; ++i)
            {
                lldb::UserSettingsControllerSP child = root->GetChildAtIndex (i);
                if (child)
                {
                    ConstString child_prefix = child->GetLevelName();
                    if (child_prefix == var_name)
                    {
                        found = true;
                        UserSettingsController::FindSettingsDescriptions (interpreter, child, current_prefix,
                                                                          var_name.AsCString(), result_stream, err);
                    }
                }
            }
            if (!found)
            {
                std::string parent_prefix;
                root->BuildParentPrefix (parent_prefix);
                err.SetErrorStringWithFormat ("Cannot find match for '%s.%s'\n", parent_prefix.c_str(), search_name);
                return;
            }
        }
    }
    else
    {
        // It must be a child name; find the child and call this function recursively on child.
        ConstString child_name (names.GetArgumentAtIndex (0));

        StreamString rest_of_search_name;
        for (int i = 0; i < num_pieces; ++i)
        {
            rest_of_search_name.Printf ("%s", names.GetArgumentAtIndex (i));
            if ((i + 1) < num_pieces)
                rest_of_search_name.Printf (".");
        }

        int num_children = root->GetNumChildren();
        bool found = false;
        for (int i = 0; i < num_children && !found; ++i)
        {
            lldb::UserSettingsControllerSP child = root->GetChildAtIndex (i);
            if (child)
            {
                ConstString child_prefix = child->GetLevelName();
                if (child_prefix == child_name)
                {
                    found = true;
                    UserSettingsController::FindSettingsDescriptions (interpreter, child, current_prefix,
                                                                      rest_of_search_name.GetData(), result_stream,
                                                                      err);
                }
            }
        }
        if (!found)
        {
            std::string parent_prefix;
            root->BuildParentPrefix (parent_prefix);
            err.SetErrorStringWithFormat ("Cannot find match for '%s.%s'\n", parent_prefix.c_str(), search_name);
            return;
        }
    }
}

void
UserSettingsController::GetAllVariableValues (CommandInterpreter &interpreter,
                                              lldb::UserSettingsControllerSP root,
                                              std::string &current_prefix,
                                              StreamString &result_stream,
                                              Error &err)
{
    StreamString description;
    int num_entries = root->m_settings.global_settings.size();
    lldb::SettableVariableType var_type;


    for (int i = 0; i < num_entries; ++i)
    {
        StreamString full_var_name;
        SettingEntry entry = root->m_settings.global_settings[i];
        if (! current_prefix.empty())
            full_var_name.Printf ("%s.%s", current_prefix.c_str(), entry.var_name);
        else
            full_var_name.Printf ("%s", entry.var_name);
        StringList value = root->GetVariable (full_var_name.GetData(), var_type, 
                                              interpreter.GetDebugger().GetInstanceName().AsCString(), err);
        description.Clear();
        if (value.GetSize() == 1)   
            description.Printf ("%s (%s) = '%s'", full_var_name.GetData(), GetTypeString (entry.var_type),
                                value.GetStringAtIndex (0));
        else
        {
            description.Printf ("%s (%s):\n", full_var_name.GetData(), GetTypeString (entry.var_type));
            for (int j = 0; j < value.GetSize(); ++j)
                if (entry.var_type == lldb::eSetVarTypeArray)
                    description.Printf ("  [%d]: '%s'\n", j, value.GetStringAtIndex (j));
                else if (entry.var_type == lldb::eSetVarTypeDictionary)
                    description.Printf ("  '%s'\n", value.GetStringAtIndex (j));
        }

        result_stream.Printf ("%s\n", description.GetData());
    }

    root->GetAllInstanceVariableValues (interpreter, result_stream);
    root->GetAllPendingSettingValues (result_stream);
    if (root->GetLevelName().GetLength() > 0)               // Don't bother with default values for Debugger level.
         root->GetAllDefaultSettingValues (result_stream);


    // Now, recurse across all children.
    int num_children = root->GetNumChildren();
    for (int i = 0; i < num_children; ++i)
    {
        lldb::UserSettingsControllerSP child = root->GetChildAtIndex (i);
        
        if (child)
        {
            ConstString child_prefix = child->GetLevelName();
            StreamString new_prefix;
            if (! current_prefix.empty())
                new_prefix.Printf ("%s.%s", current_prefix.c_str(), child_prefix.AsCString());
            else
                new_prefix.Printf ("%s", child_prefix.AsCString());
            std::string new_prefix_str = new_prefix.GetData();
            UserSettingsController::GetAllVariableValues (interpreter, child, new_prefix_str, result_stream, 
                                                          err);
        }
    }

}

Args
UserSettingsController::BreakNameIntoPieces (const char *full_dot_name)
{
    Args return_value;
    std::string name_string (full_dot_name);
    bool done = false;
    
    std::string piece;
    std::string remainder (full_dot_name);
  
    while (!done)
    {
        size_t idx = remainder.find_first_of ('.');
        piece = remainder.substr (0, idx);
        return_value.AppendArgument (piece.c_str());
        if (idx != std::string::npos)
            remainder = remainder.substr (idx+1);
        else
            done = true;
    }

    return return_value;
}

bool
UserSettingsController::IsLiveInstance (const std::string &instance_name)
{
    std::map<std::string, InstanceSettings *>::iterator pos;
    
    pos = m_live_settings.find (instance_name);
    if (pos != m_live_settings.end())
        return true;
    
    return false;
}

int
UserSettingsController::CompleteSettingsValue (lldb::UserSettingsControllerSP root_settings,
                                               const char *full_dot_name,
                                               const char *partial_value,
                                               bool &word_complete,
                                               StringList &matches)
{
    Args names = UserSettingsController::BreakNameIntoPieces (full_dot_name);
    int num_pieces = names.GetArgumentCount();
    word_complete = true;

    ConstString root_level = root_settings->GetLevelName();
    int num_extra_levels = num_pieces - 2;
    if ((num_extra_levels > 0)
        && root_level.GetLength() > 0)
    {
        ConstString current_level (names.GetArgumentAtIndex (0));
        if (current_level == root_level)
        {
            names.Shift();
            --num_extra_levels;
        }
        else
            return 0;
    }

    for (int i = 0; i < num_extra_levels; ++i)
    {
        ConstString child_level (names.GetArgumentAtIndex (0));
        bool found = false;
        int num_children = root_settings->GetNumChildren();
        for (int j = 0; j < num_children && !found; ++j)
        {
            if (root_settings->GetChildAtIndex (j)->GetLevelName() == child_level)
            {
                found = true;
                root_settings = root_settings->GetChildAtIndex (j);
                names.Shift();
            }
        }
        if (!found)
            return 0;
    }

    if (names.GetArgumentCount() != 2)
        return 0;

    std::string next_name (names.GetArgumentAtIndex (0));
    int len = next_name.length();
    names.Shift();

    if ((next_name[0] == '[') && (next_name[len-1] == ']'))
    {
        // 'next_name' is instance name.  Instance names are irrelevent here.
    }
    else
    {
        // 'next_name' is child name.
        bool found = false;
        int num_children = root_settings->GetNumChildren();
        ConstString child_level (next_name.c_str());
        for (int j = 0; j < num_children && !found; ++j)
        {
            if (root_settings->GetChildAtIndex (j)->GetLevelName() == child_level)
            {
                found = true;
                root_settings = root_settings->GetChildAtIndex (j);
            }
        }
        if (!found)
            return 0;
    }

    ConstString var_name (names.GetArgumentAtIndex(0));
    const SettingEntry *entry = root_settings->GetGlobalEntry (var_name);
    if (entry == NULL)
        entry = root_settings->GetInstanceEntry (var_name);

    if (entry == NULL)
        return 0;

    if (entry->var_type == lldb::eSetVarTypeBoolean)
        return UserSettingsController::BooleanMatches (partial_value, word_complete, matches);
    else if (entry->var_type == lldb::eSetVarTypeEnum)
        return UserSettingsController::EnumMatches (partial_value, entry->enum_values, word_complete, matches);
    else
        return 0;
}

int
UserSettingsController::BooleanMatches (const char *partial_value,
                                        bool &word_complete,
                                        StringList &matches)
{
    static const std::string true_string ("true");
    static const std::string false_string ("false");

    if (partial_value == NULL)
    {
        matches.AppendString ("true");
        matches.AppendString ("false");
    }
    else
    {
        int partial_len = strlen (partial_value);

        if ((partial_len <= true_string.length())
            && (true_string.find (partial_value) == 0))
            matches.AppendString ("true");
        else if ((partial_len <= false_string.length())
                 && (false_string.find (partial_value) == 0))
            matches.AppendString ("false");
    }

    word_complete = false;
    if (matches.GetSize() == 1)
        word_complete = true;

    return matches.GetSize();
}

int
UserSettingsController::EnumMatches (const char *partial_value,
                                     lldb::OptionEnumValueElement *enum_values,
                                     bool &word_complete,
                                     StringList &matches)
{
    int len = (partial_value != NULL) ? strlen (partial_value) : 0;

    int i = 0;
    while (enum_values[i].string_value != NULL)
    {
        if (len == 0)
            matches.AppendString (enum_values[i].string_value);
        else
        {
            std::string tmp_value (enum_values[i].string_value);
            if ((len <= tmp_value.length())
                && tmp_value.find (partial_value) == 0)
              matches.AppendString (enum_values[i].string_value);
        }
        ++i;
    }

    word_complete = false;
    if (matches.GetSize() == 1)
      word_complete = true;

    return matches.GetSize();
}

int
UserSettingsController::CompleteSettingsNames (lldb::UserSettingsControllerSP root_settings,
                                               Args &partial_setting_name_pieces,
                                               bool &word_complete,
                                               StringList &matches)
{
    int num_matches = 0;
    int num_name_pieces = partial_setting_name_pieces.GetArgumentCount();

    if (num_name_pieces > 1)
    {
        // There are at least two pieces, perhaps with multiple level names preceding them.
        // First traverse all the extra levels, until we have exactly two pieces left.
        
        int num_extra_levels = num_name_pieces - 2;

        // Deal with current level first.

        ConstString root_level = root_settings->GetLevelName();
        if ((num_extra_levels > 0)
            && (root_level.GetLength() > 0))
        {
            ConstString current_level (partial_setting_name_pieces.GetArgumentAtIndex (0));
            if (current_level == root_level)
            {
                partial_setting_name_pieces.Shift();
                --num_extra_levels;
            }
            else
                return 0; // The current level did not match the name pieces; something is wrong, so return immediately
            
        }

        for (int i = 0; i < num_extra_levels; ++i)
        {
            ConstString child_level (partial_setting_name_pieces.GetArgumentAtIndex (0));
            bool found = false;
            int num_children = root_settings->GetNumChildren();
            for (int j = 0; j < num_children && !found; ++j)
            {
                if (root_settings->GetChildAtIndex (j)->GetLevelName() == child_level)
                {
                    found = true;
                    root_settings = root_settings->GetChildAtIndex (j);
                    partial_setting_name_pieces.Shift();
                }
            }
            if (! found)
            {
                return 0; // Unable to find a matching child level name; something is wrong, so return immediately.
            }
        }

        // Now there should be exactly two name pieces left.  If not there is an error, so return immediately

        if (partial_setting_name_pieces.GetArgumentCount() != 2)
            return 0;

        std::string next_name (partial_setting_name_pieces.GetArgumentAtIndex (0));
        int len = next_name.length();
        partial_setting_name_pieces.Shift();

        if ((next_name[0] == '[') && (next_name[len-1] == ']'))
        {
            // 'next_name' is an instance name.  The last name piece must be a non-empty partial match against an
            // instance_name, assuming 'next_name' is valid.

            if (root_settings->IsLiveInstance (next_name))
            {
                std::string complete_prefix;
                root_settings->BuildParentPrefix (complete_prefix);

                num_matches = root_settings->InstanceVariableMatches(partial_setting_name_pieces.GetArgumentAtIndex(0),
                                                                     complete_prefix,
                                                                     next_name.c_str(),
                                                                     matches);
                word_complete = true;
                if (num_matches > 1)
                    word_complete = false;

                return num_matches;
            }
            else
                return 0;   // Invalid instance_name
        }
        else
        {
            // 'next_name' must be a child name.  Find the correct child and pass the remaining piece to be resolved.
            bool found = false;
            int num_children = root_settings->GetNumChildren();
            ConstString child_level (next_name.c_str());
            for (int i = 0; i < num_children; ++i)
            {
                if (root_settings->GetChildAtIndex (i)->GetLevelName() == child_level)
                {
                    found = true;
                    return UserSettingsController::CompleteSettingsNames (root_settings->GetChildAtIndex (i),
                                                                          partial_setting_name_pieces,
                                                                          word_complete, matches);
                }
            }
            if (!found)
                return 0;
        }
    }
    else if (num_name_pieces == 1)
    {
        std::string complete_prefix;
        root_settings->BuildParentPrefix (complete_prefix);

        word_complete = true;
        std::string name (partial_setting_name_pieces.GetArgumentAtIndex (0));

        if (name[0] == '[')
        {
            // It's a partial instance name.

            num_matches = root_settings->LiveInstanceMatches (name.c_str(), complete_prefix, word_complete, matches);
        }
        else
        {
            // It could be anything *except* an instance name...

            num_matches = root_settings->GlobalVariableMatches (name.c_str(), complete_prefix, matches);
            num_matches += root_settings->InstanceVariableMatches (name.c_str(), complete_prefix, NULL, matches);
            num_matches += root_settings->ChildMatches (name.c_str(), complete_prefix, word_complete, matches);
        }

        if (num_matches > 1)
            word_complete = false;

        return num_matches;
    }
    else
    {
        // We have a user settings controller with a blank partial string.  Return everything possible at this level.

        std::string complete_prefix;
        root_settings->BuildParentPrefix (complete_prefix);
        num_matches = root_settings->GlobalVariableMatches (NULL, complete_prefix, matches);
        num_matches += root_settings->InstanceVariableMatches (NULL, complete_prefix, NULL, matches);
        num_matches += root_settings->LiveInstanceMatches (NULL, complete_prefix, word_complete, matches);
        num_matches += root_settings->ChildMatches (NULL, complete_prefix, word_complete, matches);
        word_complete = false;
        return num_matches;
    }

    return num_matches;
}

int
UserSettingsController::GlobalVariableMatches (const char *partial_name,
                                               const std::string &complete_prefix,
                                               StringList &matches)
{
    int partial_len = (partial_name != NULL) ? strlen (partial_name) : 0;
    int num_matches = 0;

    for (size_t i = 0; i < m_settings.global_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.global_settings[i];
        std::string var_name (entry.var_name);
        if ((partial_len == 0)
            || ((partial_len <= var_name.length())
                && (var_name.find (partial_name) == 0)))
        {
            StreamString match_name;
            if (complete_prefix.length() > 0)
            {
                match_name.Printf ("%s.%s", complete_prefix.c_str(), var_name.c_str());
                matches.AppendString (match_name.GetData());
            }
            else
                matches.AppendString (var_name.c_str());
            ++num_matches;
        }
    }
    return num_matches;
}

int
UserSettingsController::InstanceVariableMatches (const char *partial_name,
                                                 const std::string &complete_prefix,
                                                 const char *instance_name,
                                                 StringList &matches)
{
    int partial_len = (partial_name != NULL) ? strlen (partial_name) : 0;
    int num_matches = 0;

    for (size_t i = 0; i < m_settings.instance_settings.size(); ++i)
    {
        SettingEntry &entry = m_settings.instance_settings[i];
        std::string var_name (entry.var_name);
        if ((partial_len == 0)
            || ((partial_len <= var_name.length())
                && (var_name.find (partial_name) == 0)))
        {
            StreamString match_name;
            if (complete_prefix.length() > 0)
            {
                if (instance_name != NULL)
                    match_name.Printf ("%s.%s.%s", complete_prefix.c_str(), instance_name, var_name.c_str());
                else
                    match_name.Printf ("%s.%s", complete_prefix.c_str(), var_name.c_str());

                matches.AppendString (match_name.GetData());
            }
            else
            {
                if (instance_name != NULL)
                {
                    match_name.Printf ("%s.%s", instance_name, var_name.c_str());
                    matches.AppendString (match_name.GetData());
                }
                else
                    matches.AppendString (var_name.c_str());
            }
            ++num_matches;
        }
    }
    return num_matches;
}

int
UserSettingsController::LiveInstanceMatches (const char *partial_name,
                                             const std::string &complete_prefix,
                                             bool &word_complete,
                                             StringList &matches)
{
    int partial_len = (partial_name != NULL) ? strlen (partial_name) : 0;
    int num_matches = 0;

    std::map<std::string, InstanceSettings*>::iterator pos;
    for (pos = m_live_settings.begin(); pos != m_live_settings.end(); ++pos)
    {
        std::string instance_name = pos->first;
        if ((partial_len == 0)
            || ((partial_len <= instance_name.length())
                && (instance_name.find (partial_name) == 0)))
        {
            StreamString match_name;
            if (complete_prefix.length() > 0)
                match_name.Printf ("%s.%s.", complete_prefix.c_str(), instance_name.c_str());
            else
                match_name.Printf ("%s.", instance_name.c_str());
            matches.AppendString (match_name.GetData());
            ++num_matches;
        }
    }

    if (num_matches > 0)
        word_complete = false;

    return num_matches;
}

int
UserSettingsController::ChildMatches (const char *partial_name,
                                      const std::string &complete_prefix,
                                      bool &word_complete,
                                      StringList &matches)
{
    int partial_len = (partial_name != NULL) ? strlen (partial_name) : 0;
    int num_children = GetNumChildren();
    int num_matches = 0;
    for (int i = 0; i < num_children; ++i)
    {
        std::string child_name (GetChildAtIndex(i)->GetLevelName().AsCString());
        StreamString match_name;
        if ((partial_len == 0)
          || ((partial_len <= child_name.length())
              && (child_name.find (partial_name) == 0)))
        {
            if (complete_prefix.length() > 0)
                match_name.Printf ("%s.%s.", complete_prefix.c_str(), child_name.c_str());
            else
                match_name.Printf ("%s.", child_name.c_str());
            matches.AppendString (match_name.GetData());
            ++num_matches;
        }
    }

    if (num_matches > 0)
        word_complete = false;

    return num_matches;
}

void
UserSettingsController::VerifyOperationForType (lldb::SettableVariableType var_type, 
                                                lldb::VarSetOperationType op, 
                                                const ConstString &var_name,
                                                Error &err)
{
    if (op == lldb::eVarSetOperationAssign)
        return;


    if (op == lldb::eVarSetOperationInvalid)
    {
        err.SetErrorString ("Invalid 'settings ' subcommand operation.\n");  
        return;
    }

    switch (op)
    {
        case lldb::eVarSetOperationInsertBefore:
        case lldb::eVarSetOperationInsertAfter:
            if (var_type != lldb::eSetVarTypeArray)
                err.SetErrorString ("Invalid operation:  This operation can only be performed on array variables.\n");
            break;
        case lldb::eVarSetOperationReplace:
        case lldb::eVarSetOperationRemove:
            if ((var_type != lldb::eSetVarTypeArray)
                && (var_type != lldb::eSetVarTypeDictionary))
                err.SetErrorString ("Invalid operation:  This operation can only be performed on array or dictionary"
                                    " variables.\n");
            break;
        case lldb::eVarSetOperationAppend:
        case lldb::eVarSetOperationClear:
            if ((var_type != lldb::eSetVarTypeArray)
                && (var_type != lldb::eSetVarTypeDictionary)
                && (var_type != lldb::eSetVarTypeString))
                err.SetErrorString ("Invalid operation:  This operation can only be performed on array, dictionary "
                                    "or string variables.\n");
            break;
        default:
            break;
    }

    return;
}

void
UserSettingsController::UpdateStringVariable (lldb::VarSetOperationType op, 
                                              std::string &string_var, 
                                              const char *new_value,
                                              Error &err)
{
    if (op == lldb::eVarSetOperationAssign)
    {
        if (new_value && new_value[0])
            string_var.assign (new_value);
        else
            string_var.clear();
    }
    else if (op == lldb::eVarSetOperationAppend)
    {
        if (new_value && new_value[0])
            string_var.append (new_value);
    }
    else if (op == lldb::eVarSetOperationClear)
        string_var.clear();
    else
        err.SetErrorString ("Unrecognized operation. Cannot update value.\n");
}

void
UserSettingsController::UpdateBooleanVariable (lldb::VarSetOperationType op,
                                               bool &bool_var,
                                               const char *new_value,
                                               Error &err)
{
    if (op != lldb::eVarSetOperationAssign)
        err.SetErrorString ("Invalid operation for Boolean variable.  Cannot update value.\n");

    if (new_value && new_value[0])
    {
        if ((::strcasecmp(new_value, "true") == 0) ||
            (::strcasecmp(new_value, "yes") == 0)  ||
            (::strcasecmp(new_value, "on") == 0)   ||
            (::strcasecmp(new_value, "1") == 0))
            bool_var = true;
        else 
        if ((::strcasecmp(new_value, "false") == 0) ||
            (::strcasecmp(new_value, "no") == 0)    ||
            (::strcasecmp(new_value, "off") == 0)   ||
            (::strcasecmp(new_value, "0") == 0))
            bool_var = false;
        else
            err.SetErrorStringWithFormat ("Invalid boolean value '%s'\n", new_value);
    }
    else
        err.SetErrorString ("Invalid value.  Cannot perform update.\n");

}

void
UserSettingsController::UpdateStringArrayVariable (lldb::VarSetOperationType op, 
                                                   const char *index_value,
                                                   Args &array_var,
                                                   const char *new_value,
                                                   Error &err)
{
    int index = -1;
    bool valid_index = true;
    
    if (index_value != NULL)
    {
        for (int i = 0; i < strlen(index_value); ++i)
            if (!isdigit (index_value[i]))
            {
                valid_index = false;
                err.SetErrorStringWithFormat ("'%s' is not a valid integer index. Cannot update array value.\n", 
                                              index_value);
            }
                
        if (valid_index)
            index = atoi (index_value);
            
        if (index < 0
            || index >= array_var.GetArgumentCount())
        {
            valid_index = false;
            err.SetErrorStringWithFormat ("%d is outside the bounds of the specified array variable. "
                                          "Cannot update array value.\n", index);
        }
    }

    switch (op) 
    {
        case lldb::eVarSetOperationAssign:
            array_var.SetCommandString (new_value);
            break;
        case lldb::eVarSetOperationReplace:
        {
            if (valid_index)
                array_var.ReplaceArgumentAtIndex (index, new_value);
            break;
        }
        case lldb::eVarSetOperationInsertBefore:
        case lldb::eVarSetOperationInsertAfter:
        {
            if (valid_index)
            {
                Args new_array (new_value);
                if (op == lldb::eVarSetOperationInsertAfter)
                    ++index;
                for (int i = 0; i < new_array.GetArgumentCount(); ++i)
                    array_var.InsertArgumentAtIndex (index, new_array.GetArgumentAtIndex (i));
            }
            break;
        }
        case lldb::eVarSetOperationRemove:
        {
            if (valid_index)
                array_var.DeleteArgumentAtIndex (index);
            break;
        }
        case lldb::eVarSetOperationAppend:
        {
            Args new_array (new_value);
            array_var.AppendArguments (new_array);
            break;
        }
        case lldb::eVarSetOperationClear:
            array_var.Clear();
            break;
        default:
            err.SetErrorString ("Unrecognized operation. Cannot update value.\n");
            break;
    }
}

void
UserSettingsController::UpdateDictionaryVariable (lldb::VarSetOperationType op,
                                                  const char *index_value,
                                                  std::map<std::string, std::string> &dictionary,
                                                  const char *new_value,
                                                  Error &err)
{
    switch (op)
    {
        case lldb::eVarSetOperationReplace:
            if (index_value != NULL)
            {
                std::string key (index_value);
                std::map<std::string, std::string>::iterator pos;
                
                pos = dictionary.find (key);
                if (pos != dictionary.end())
                    dictionary[key] = new_value;
                else
                    err.SetErrorStringWithFormat ("'%s' is not an existing key; cannot replace value.\n", index_value);
            }
            else
                err.SetErrorString ("'settings replace' requires a key for dictionary variables. No key supplied.\n");
            break;
        case lldb::eVarSetOperationRemove:
            if (index_value != NULL)
            {
                std::string key (index_value);
                dictionary.erase (key);
            }
            else
                err.SetErrorString ("'settings remove' requires a key for dictionary variables. No key supplied.\n");
            break;
        case lldb::eVarSetOperationClear:
            dictionary.clear ();
            break;
        case lldb::eVarSetOperationAppend:
        case lldb::eVarSetOperationAssign:
            {
                // Clear the dictionary if it's an assign with new_value as NULL.
                if (new_value == NULL && op == lldb::eVarSetOperationAssign)
                {
                    dictionary.clear ();
                    break;
                }
                Args args (new_value);
                size_t num_args = args.GetArgumentCount();
                RegularExpression regex("(\\[\"?)?"                 // Regex match 1 (optional key prefix of '["' pr '[')
                                        "([A-Za-z_][A-Za-z_0-9]*)"  // Regex match 2 (key string)
                                        "(\"?\\])?"                 // Regex match 3 (optional key suffix of '"]' pr ']')
                                        "="                         // The equal sign that is required
                                        "(.*)");                    // Regex match 4 (value string)
                std::string key, value;

                for (size_t i = 0; i < num_args; ++i)
                {
                    const char *key_equal_value_arg = args.GetArgumentAtIndex (i);
                    // Execute the regular expression on each arg.
                    if (regex.Execute(key_equal_value_arg, 5))
                    {
                        // The regular expression succeeded. The match at index
                        // zero will be the entire string that matched the entire
                        // regular expression. The match at index 1 - 4 will be
                        // as mentioned above by the creation of the regex pattern.
                        // Match index 2 is the key, match index 4 is the value.
                        regex.GetMatchAtIndex (key_equal_value_arg, 2, key);
                        regex.GetMatchAtIndex (key_equal_value_arg, 4, value);
                        dictionary[key] = value;
                    }
                    else
                    {
                        err.SetErrorString ("Invalid format for dictionary value.  Expected one of '[\"<key>\"]=<value>', '[<key>]=<value>', or '<key>=<value>'\n");
                    }
                }
            }  
            break;
        case lldb::eVarSetOperationInsertBefore:
        case lldb::eVarSetOperationInsertAfter:
            err.SetErrorString ("Specified operation cannot be performed on dictionary variables.\n");
            break;
        default:
            err.SetErrorString ("Unrecognized operation.\n");
            break;
    }
}

const char *
UserSettingsController::EnumToString (const lldb::OptionEnumValueElement *enum_values,
                                      int value)
{
    int i = 0;
    while (enum_values[i].string_value != NULL)
    {
        if (enum_values[i].value == value)
            return enum_values[i].string_value;
        ++i;
    }

    return "Invalid enumeration value";
}


void
UserSettingsController::UpdateEnumVariable (lldb::OptionEnumValueElement *enum_values,
                                            int *enum_var,
                                            const char *new_value,
                                            Error &err)
{
    bool found_one;
    
    *enum_var = Args::StringToOptionEnum (new_value, enum_values, enum_values[0].value, &found_one);

    if (!found_one)
        err.SetErrorString ("Invalid enumeration value; cannot update variable.\n");
}

void
UserSettingsController::RenameInstanceSettings (const char *old_name, const char *new_name)
{
    Mutex::Locker live_mutex (m_live_settings_mutex);
    Mutex::Locker pending_mutex (m_pending_settings_mutex);
    std::string old_name_key (old_name);
    std::string new_name_key (new_name);

    // First, find the live instance settings for the old_name.  If they don't exist in the live settings
    // list, then this is not a setting that can be renamed.

    if ((old_name_key[0] != '[') || (old_name_key[old_name_key.size() -1] != ']'))
    {
        StreamString tmp_str;
        tmp_str.Printf ("[%s]", old_name);
          old_name_key = tmp_str.GetData();
    }

    if ((new_name_key[0] != '[') || (new_name_key[new_name_key.size() -1] != ']'))
    {
        StreamString tmp_str;
        tmp_str.Printf ("[%s]", new_name);
        new_name_key = tmp_str.GetData();
    }

    if (old_name_key.compare (new_name_key) == 0) 
        return;

    size_t len = new_name_key.length();
    std::string stripped_new_name = new_name_key.substr (1, len-2);  // new name without the '[ ]'

    std::map<std::string, InstanceSettings *>::iterator pos;

    pos = m_live_settings.find (old_name_key);
    if (pos != m_live_settings.end())
    {
        InstanceSettings *live_settings = pos->second;

        // Rename the settings.
        live_settings->ChangeInstanceName (stripped_new_name);

        // Now see if there are any pending settings for the new name; if so, copy them into live_settings.
        std::map<std::string,  lldb::InstanceSettingsSP>::iterator pending_pos;
        pending_pos = m_pending_settings.find (new_name_key);
        if (pending_pos != m_pending_settings.end())
        {
            lldb::InstanceSettingsSP pending_settings_sp = pending_pos->second;
            live_settings->CopyInstanceSettings (pending_settings_sp, false);
        }

        // Erase the old entry (under the old name) from live settings.
        m_live_settings.erase (pos);

        // Add the new entry, with the new name, into live settings.
        m_live_settings[new_name_key] = live_settings;
    }
}

//----------------------------------------------------------------------
// class InstanceSettings
//----------------------------------------------------------------------

InstanceSettings::InstanceSettings (UserSettingsController &owner, const char *instance_name, bool live_instance) :
    m_owner (owner),
    m_instance_name (instance_name)
{
    if ((m_instance_name != InstanceSettings::GetDefaultName())
        && (m_instance_name !=  InstanceSettings::InvalidName())
        && live_instance)
        m_owner.RegisterInstanceSettings (this);
}

InstanceSettings::~InstanceSettings ()
{
    if (m_instance_name != InstanceSettings::GetDefaultName())
        m_owner.UnregisterInstanceSettings (this);
}

const ConstString &
InstanceSettings::GetDefaultName ()
{
    static const ConstString g_default_settings_name ("[DEFAULT]");

    return g_default_settings_name;
}

const ConstString &
InstanceSettings::InvalidName ()
{
    static const ConstString g_invalid_name ("Invalid instance name");

    return g_invalid_name;
}

void
InstanceSettings::ChangeInstanceName (const std::string &new_instance_name)
{
    m_instance_name.SetCString (new_instance_name.c_str());
}



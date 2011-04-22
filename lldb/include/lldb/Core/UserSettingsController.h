//====-- UserSettingsController.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UserSettingsController_h_
#define liblldb_UserSettingsController_h_

// C Includes
// C++ Includes

#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

typedef struct
{
    const char *var_name;
    SettableVariableType var_type;
    const char *default_value;
    OptionEnumValueElement *enum_values;
    bool initialized;
    bool hidden;
    const char *description;   //help text
} SettingEntry;


typedef struct
{
    lldb::UserSettingsControllerSP parent;
    ConstString level_name;
    std::vector<SettingEntry> global_settings;
    std::vector<SettingEntry> instance_settings;
} UserSettingDefinition;

class UserSettingsController
{
public:

    UserSettingsController (const char *level_name, 
                            const lldb::UserSettingsControllerSP &parent);

    virtual
    ~UserSettingsController ();

    // Pure virtual functions, which all sub-classes must implement.
    virtual lldb::InstanceSettingsSP
    CreateInstanceSettings (const char *instance_name) = 0;

    // Virtual functions that you can override if you have global settings
    // (not instance specific).
    virtual bool
    SetGlobalVariable (const ConstString &var_name,
                       const char *index_value,
                       const char *value,
                       const SettingEntry &entry,
                       const VarSetOperationType op,
                       Error &err);

    virtual bool
    GetGlobalVariable (const ConstString &var_name, 
                       StringList &value,
                       Error &err);
    
    // End of pure virtual functions.
    StringList
    GetVariable (const char *full_dot_name, 
                 SettableVariableType &var_type,
                 const char *debugger_instance_name,
                 Error &err);

    Error
    SetVariable (const char *full_dot_name, 
                 const char *value, 
                 const VarSetOperationType op,
                 const bool override,
                 const char *debugger_instance_name,
                 const char *index_value = NULL);

    const lldb::UserSettingsControllerSP &
    GetParent ();

    const ConstString &
    GetLevelName ();

    void
    RegisterChild (const lldb::UserSettingsControllerSP &child);

    void
    RemoveChild (const lldb::UserSettingsControllerSP &child);

    void
    CreateSettingsVector (const SettingEntry *table,
                          const bool global);

    void
    CreateDefaultInstanceSettings ();

    void
    InitializeGlobalVariables ();

    const lldb::InstanceSettingsSP &
    FindPendingSettings (const ConstString &instance_name);

    void
    RemovePendingSettings (const ConstString &instance_name);
  
    void
    RegisterInstanceSettings (InstanceSettings *instance_settings);

    void
    UnregisterInstanceSettings (InstanceSettings *instance_settings);

    void
    RenameInstanceSettings (const char *old_name, const char *new_name);

    // -------------------------------------------------------------------------
    // Public static methods
    // -------------------------------------------------------------------------

    static void
    FindAllSettingsDescriptions (CommandInterpreter &interpreter,
                                 const lldb::UserSettingsControllerSP& usc_sp, 
                                 const char *current_prefix, 
                                 Stream &result_stream,
                                 Error &err);

    static void
    FindSettingsDescriptions (CommandInterpreter &interpreter,
                              const lldb::UserSettingsControllerSP& usc_sp, 
                              const char *current_prefix, 
                              const char *search_name,
                              Stream &result_stream,
                              Error &err);
    
    static void
    SearchAllSettingsDescriptions (CommandInterpreter &interpreter,
                                   const lldb::UserSettingsControllerSP& usc_sp,
                                   const char *current_prefix,
                                   const char *search_word,
                                   Stream &result_stream);

    static void
    GetAllVariableValues (CommandInterpreter &interpreter,
                          const lldb::UserSettingsControllerSP& usc_sp,
                          const char *current_prefix, 
                          Stream &result_stream,
                          Error &err);

    static bool
    DumpValue (CommandInterpreter &interpreter, 
               const lldb::UserSettingsControllerSP& usc_sp,
               const char *variable_dot_name,
               Stream &strm);
    
    static bool
    DumpValue (const char *variable_dot_name,
               SettableVariableType var_type,
               const StringList &variable_value,
               Stream &strm);

    static int
    CompleteSettingsNames (const lldb::UserSettingsControllerSP& usc_sp,
                           Args &partial_setting_name_pieces,
                           bool &word_complete,
                           StringList &matches);

    static int
    CompleteSettingsValue (const lldb::UserSettingsControllerSP& usc_sp,
                           const char *full_dot_name,
                           const char *partial_value,
                           bool &word_complete,
                           StringList &matches);

    static Args
    BreakNameIntoPieces (const char *full_dot_name);

    static const char *
    GetTypeString (SettableVariableType var_type);


    static const char *
    EnumToString (const OptionEnumValueElement *enum_values, int value);

    static void
    UpdateStringVariable (VarSetOperationType op, 
                          std::string &string_var, 
                          const char *new_value,
                          Error &err);


    static void
    UpdateBooleanVariable (VarSetOperationType op,
                           bool &bool_var,
                           const char *new_value,
                           bool clear_value, // Used for op == eVarSetOperationClear
                           Error &err);

    static void
    UpdateStringArrayVariable (VarSetOperationType op, 
                               const char *index_value, 
                               Args &array_var, 
                               const char *new_value,
                               Error &err);
  
    static void
    UpdateDictionaryVariable (VarSetOperationType op,
                              const char *index_value,
                              std::map<std::string, std::string> &dictionary,
                              const char *new_value,
                              Error &err);

    static void
    UpdateEnumVariable (OptionEnumValueElement *enum_values,
                        int *enum_var,
                        const char *new_value,
                        Error &err);

    static bool
    InitializeSettingsController (lldb::UserSettingsControllerSP &controller_sp,
                                  SettingEntry *global_settings,
                                  SettingEntry *instance_settings);

    static void
    FinalizeSettingsController (lldb::UserSettingsControllerSP &controller_sp);


protected:

    // -------------------------------------------------------------------------
    // Protected methods are declared below here.
    // -------------------------------------------------------------------------

    bool
    IsLiveInstance (const std::string &instance_name);

    int
    GlobalVariableMatches (const char *partial_name,
                           const std::string &complete_prefix,
                           StringList &matches);

    int
    InstanceVariableMatches (const char *partial_name,
                             const std::string &complete_prefix,
                             const char *instance_name,
                             StringList &matches);

    int
    LiveInstanceMatches (const char *partial_name,
                         const std::string &complete_prefix,
                         bool &word_complete,
                         StringList &matches);

    int
    ChildMatches (const char *partial_name,
                  const std::string &complete_prefix,
                  bool &word_complete,
                  StringList &matches);


    size_t
    GetNumChildren ();

    const lldb::UserSettingsControllerSP
    GetChildAtIndex (size_t index);


    const SettingEntry *
    GetGlobalEntry (const ConstString &var_name);

    const SettingEntry *
    GetInstanceEntry (const ConstString &var_name);

    void
    BuildParentPrefix (std::string &parent_prefix);


    void
    CopyDefaultSettings (const lldb::InstanceSettingsSP &new_settings,
                         const ConstString &instance_name,
                         bool pending);

    lldb::InstanceSettingsSP
    PendingSettingsForInstance (const ConstString &instance_name);

    InstanceSettings *
    FindSettingsForInstance (const ConstString &instance_name);

    void
    GetAllPendingSettingValues (Stream &result_stream);

    void
    GetAllDefaultSettingValues (Stream &result_stream);

    void
    GetAllInstanceVariableValues (CommandInterpreter &interpreter, 
                                  Stream &result_stream);

    void
    OverrideAllInstances (const ConstString &var_name, 
                          const char *value,
                          VarSetOperationType op, 
                          const char *index_value, 
                          Error &err);

    UserSettingDefinition &
    GetControllerSettings () { return m_settings; }

    // -------------------------------------------------------------------------
    // Static protected methods are declared below here.
    // -------------------------------------------------------------------------

    static void
    PrintEnumValues (const OptionEnumValueElement *enum_values, 
                     Stream &str);
    

    static int
    BooleanMatches (const char *partial_value,
                    bool &word_complete,
                    StringList &matches);
    
    static int
    EnumMatches (const char *partial_value,
                 OptionEnumValueElement *enum_values,
                 bool &word_complete,
                 StringList &matches);

    static void
    VerifyOperationForType (SettableVariableType var_type, 
                            VarSetOperationType op, 
                            const ConstString &var_name,
                            Error &err);

    // This is protected rather than private so that classes that inherit from UserSettingsController can access it.

    lldb::InstanceSettingsSP m_default_settings;

private:

    UserSettingDefinition m_settings;
    
    typedef std::map<std::string,InstanceSettings*> InstanceSettingsMap;

    std::vector<lldb::UserSettingsControllerSP> m_children;
    std::map <std::string, lldb::InstanceSettingsSP> m_pending_settings;
    InstanceSettingsMap m_live_settings;    // live settings should never be NULL (hence 'live')
    mutable Mutex m_children_mutex;
    mutable Mutex m_pending_settings_mutex;
    mutable Mutex m_live_settings_mutex;

    DISALLOW_COPY_AND_ASSIGN (UserSettingsController);
};

class InstanceSettings 
{
public:

    InstanceSettings (UserSettingsController &owner, const char *instance_name, bool live_instance = true);

    InstanceSettings (const InstanceSettings &rhs);

    virtual
    ~InstanceSettings ();

    InstanceSettings&
    operator= (const InstanceSettings &rhs);

    // Begin Pure Virtual Functions

    virtual void
    UpdateInstanceSettingsVariable (const ConstString &var_name,
                                    const char *index_value,
                                    const char *value,
                                    const ConstString &instance_name,
                                    const SettingEntry &entry,
                                    VarSetOperationType op,
                                    Error &err,
                                    bool pending) = 0;

    virtual bool
    GetInstanceSettingsValue (const SettingEntry &entry,
                              const ConstString &var_name,
                              StringList &value,
                              Error *err) = 0;

    virtual void
    CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                          bool pending) = 0;

    virtual const ConstString
    CreateInstanceName () = 0;

    // End Pure Virtual Functions

    const ConstString &
    GetInstanceName () { return m_instance_name; }


    void
    ChangeInstanceName (const std::string &new_instance_name);

    static const ConstString &
    GetDefaultName ();

    static const ConstString &
    InvalidName ();

protected:

    UserSettingsController &m_owner;
    ConstString m_instance_name;
};



} // namespace lldb_private

#endif // liblldb_UserSettingsController_h_

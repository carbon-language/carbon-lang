//===-- PluginManager.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"

#include <string>
#include <vector>

using namespace lldb_private;

enum PluginAction
{
    ePluginRegisterInstance,
    ePluginUnregisterInstance,
    ePluginGetInstanceAtIndex
};


#pragma mark ABI


struct ABIInstance
{
    ABIInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ABICreateInstance create_callback;
};

typedef std::vector<ABIInstance> ABIInstances;

static bool
AccessABIInstances (PluginAction action, ABIInstance &instance, uint32_t index)
{
    static ABIInstances g_plugin_instances;

    switch (action)
    {
        case ePluginRegisterInstance:
            if (instance.create_callback)
            {
                g_plugin_instances.push_back (instance);
                return true;
            }
            break;

        case ePluginUnregisterInstance:
            if (instance.create_callback)
            {
                ABIInstances::iterator pos, end = g_plugin_instances.end();
                for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
                {
                    if (pos->create_callback == instance.create_callback)
                    {
                        g_plugin_instances.erase(pos);
                        return true;
                    }
                }
            }
            break;

        case ePluginGetInstanceAtIndex:
            if (index < g_plugin_instances.size())
            {
                instance = g_plugin_instances[index];
                return true;
            }
            break;

        default:
            break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
    (
        const char *name,
        const char *description,
        ABICreateInstance create_callback
     )
{
    if (create_callback)
    {
        ABIInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessABIInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ABICreateInstance create_callback)
{
    if (create_callback)
    {
        ABIInstance instance;
        instance.create_callback = create_callback;
        return AccessABIInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ABICreateInstance
PluginManager::GetABICreateCallbackAtIndex (uint32_t idx)
{
    ABIInstance instance;
    if (AccessABIInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

ABICreateInstance
PluginManager::GetABICreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ABIInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessABIInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}


#pragma mark Disassembler


struct DisassemblerInstance
{
    DisassemblerInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    DisassemblerCreateInstance create_callback;
};

typedef std::vector<DisassemblerInstance> DisassemblerInstances;

static bool
AccessDisassemblerInstances (PluginAction action, DisassemblerInstance &instance, uint32_t index)
{
    static DisassemblerInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            DisassemblerInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    DisassemblerCreateInstance create_callback
)
{
    if (create_callback)
    {
        DisassemblerInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessDisassemblerInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (DisassemblerCreateInstance create_callback)
{
    if (create_callback)
    {
        DisassemblerInstance instance;
        instance.create_callback = create_callback;
        return AccessDisassemblerInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackAtIndex (uint32_t idx)
{
    DisassemblerInstance instance;
    if (AccessDisassemblerInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        DisassemblerInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessDisassemblerInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}



#pragma mark DynamicLoader


struct DynamicLoaderInstance
{
    DynamicLoaderInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    DynamicLoaderCreateInstance create_callback;
};

typedef std::vector<DynamicLoaderInstance> DynamicLoaderInstances;

static bool
AccessDynamicLoaderInstances (PluginAction action, DynamicLoaderInstance &instance, uint32_t index)
{
    static DynamicLoaderInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            DynamicLoaderInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    DynamicLoaderCreateInstance create_callback
)
{
    if (create_callback)
    {
        DynamicLoaderInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessDynamicLoaderInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (DynamicLoaderCreateInstance create_callback)
{
    if (create_callback)
    {
        DynamicLoaderInstance instance;
        instance.create_callback = create_callback;
        return AccessDynamicLoaderInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackAtIndex (uint32_t idx)
{
    DynamicLoaderInstance instance;
    if (AccessDynamicLoaderInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        DynamicLoaderInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessDynamicLoaderInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}




#pragma mark ObjectFile

struct ObjectFileInstance
{
    ObjectFileInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ObjectFileCreateInstance create_callback;
};

typedef std::vector<ObjectFileInstance> ObjectFileInstances;

static bool
AccessObjectFileInstances (PluginAction action, ObjectFileInstance &instance, uint32_t index)
{
    static ObjectFileInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            ObjectFileInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ObjectFileCreateInstance create_callback
)
{
    if (create_callback)
    {
        ObjectFileInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessObjectFileInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ObjectFileCreateInstance create_callback)
{
    if (create_callback)
    {
        ObjectFileInstance instance;
        instance.create_callback = create_callback;
        return AccessObjectFileInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ObjectFileCreateInstance
PluginManager::GetObjectFileCreateCallbackAtIndex (uint32_t idx)
{
    ObjectFileInstance instance;
    if (AccessObjectFileInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}
ObjectFileCreateInstance
PluginManager::GetObjectFileCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ObjectFileInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessObjectFileInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}



#pragma mark ObjectContainer

struct ObjectContainerInstance
{
    ObjectContainerInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ObjectContainerCreateInstance create_callback;
};

typedef std::vector<ObjectContainerInstance> ObjectContainerInstances;

static bool
AccessObjectContainerInstances (PluginAction action, ObjectContainerInstance &instance, uint32_t index)
{
    static ObjectContainerInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            ObjectContainerInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ObjectContainerCreateInstance create_callback
)
{
    if (create_callback)
    {
        ObjectContainerInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessObjectContainerInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ObjectContainerCreateInstance create_callback)
{
    if (create_callback)
    {
        ObjectContainerInstance instance;
        instance.create_callback = create_callback;
        return AccessObjectContainerInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ObjectContainerCreateInstance
PluginManager::GetObjectContainerCreateCallbackAtIndex (uint32_t idx)
{
    ObjectContainerInstance instance;
    if (AccessObjectContainerInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}
ObjectContainerCreateInstance
PluginManager::GetObjectContainerCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ObjectContainerInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessObjectContainerInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}

#pragma mark LogChannel

struct LogChannelInstance
{
    LogChannelInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    LogChannelCreateInstance create_callback;
};

typedef std::vector<LogChannelInstance> LogChannelInstances;

static bool
AccessLogChannelInstances (PluginAction action, LogChannelInstance &instance, uint32_t index)
{
    static LogChannelInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            LogChannelInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    LogChannelCreateInstance create_callback
)
{
    if (create_callback)
    {
        LogChannelInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessLogChannelInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (LogChannelCreateInstance create_callback)
{
    if (create_callback)
    {
        LogChannelInstance instance;
        instance.create_callback = create_callback;
        return AccessLogChannelInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

const char *
PluginManager::GetLogChannelCreateNameAtIndex (uint32_t idx)
{
    LogChannelInstance instance;
    if (AccessLogChannelInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.name.c_str();
    return NULL;
}


LogChannelCreateInstance
PluginManager::GetLogChannelCreateCallbackAtIndex (uint32_t idx)
{
    LogChannelInstance instance;
    if (AccessLogChannelInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

LogChannelCreateInstance
PluginManager::GetLogChannelCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        LogChannelInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessLogChannelInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}

#pragma mark Process

struct ProcessInstance
{
    ProcessInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ProcessCreateInstance create_callback;
};

typedef std::vector<ProcessInstance> ProcessInstances;

static bool
AccessProcessInstances (PluginAction action, ProcessInstance &instance, uint32_t index)
{
    static ProcessInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            ProcessInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ProcessCreateInstance create_callback
)
{
    if (create_callback)
    {
        ProcessInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessProcessInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ProcessCreateInstance create_callback)
{
    if (create_callback)
    {
        ProcessInstance instance;
        instance.create_callback = create_callback;
        return AccessProcessInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ProcessCreateInstance
PluginManager::GetProcessCreateCallbackAtIndex (uint32_t idx)
{
    ProcessInstance instance;
    if (AccessProcessInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

ProcessCreateInstance
PluginManager::GetProcessCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ProcessInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessProcessInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}

#pragma mark SymbolFile

struct SymbolFileInstance
{
    SymbolFileInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    SymbolFileCreateInstance create_callback;
};

typedef std::vector<SymbolFileInstance> SymbolFileInstances;

static bool
AccessSymbolFileInstances (PluginAction action, SymbolFileInstance &instance, uint32_t index)
{
    static SymbolFileInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            SymbolFileInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    SymbolFileCreateInstance create_callback
)
{
    if (create_callback)
    {
        SymbolFileInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessSymbolFileInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (SymbolFileCreateInstance create_callback)
{
    if (create_callback)
    {
        SymbolFileInstance instance;
        instance.create_callback = create_callback;
        return AccessSymbolFileInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

SymbolFileCreateInstance
PluginManager::GetSymbolFileCreateCallbackAtIndex (uint32_t idx)
{
    SymbolFileInstance instance;
    if (AccessSymbolFileInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}
SymbolFileCreateInstance
PluginManager::GetSymbolFileCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        SymbolFileInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessSymbolFileInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}



#pragma mark SymbolVendor

struct SymbolVendorInstance
{
    SymbolVendorInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    SymbolVendorCreateInstance create_callback;
};

typedef std::vector<SymbolVendorInstance> SymbolVendorInstances;

static bool
AccessSymbolVendorInstances (PluginAction action, SymbolVendorInstance &instance, uint32_t index)
{
    static SymbolVendorInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            SymbolVendorInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    SymbolVendorCreateInstance create_callback
)
{
    if (create_callback)
    {
        SymbolVendorInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessSymbolVendorInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (SymbolVendorCreateInstance create_callback)
{
    if (create_callback)
    {
        SymbolVendorInstance instance;
        instance.create_callback = create_callback;
        return AccessSymbolVendorInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

SymbolVendorCreateInstance
PluginManager::GetSymbolVendorCreateCallbackAtIndex (uint32_t idx)
{
    SymbolVendorInstance instance;
    if (AccessSymbolVendorInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

SymbolVendorCreateInstance
PluginManager::GetSymbolVendorCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        SymbolVendorInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessSymbolVendorInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}


#pragma mark UnwindAssemblyProfiler

struct UnwindAssemblyProfilerInstance
{
    UnwindAssemblyProfilerInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    UnwindAssemblyProfilerCreateInstance create_callback;
};

typedef std::vector<UnwindAssemblyProfilerInstance> UnwindAssemblyProfilerInstances;

static bool
AccessUnwindAssemblyProfilerInstances (PluginAction action, UnwindAssemblyProfilerInstance &instance, uint32_t index)
{
    static UnwindAssemblyProfilerInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            UnwindAssemblyProfilerInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    UnwindAssemblyProfilerCreateInstance create_callback
)
{
    if (create_callback)
    {
        UnwindAssemblyProfilerInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessUnwindAssemblyProfilerInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (UnwindAssemblyProfilerCreateInstance create_callback)
{
    if (create_callback)
    {
        UnwindAssemblyProfilerInstance instance;
        instance.create_callback = create_callback;
        return AccessUnwindAssemblyProfilerInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

UnwindAssemblyProfilerCreateInstance
PluginManager::GetUnwindAssemblyProfilerCreateCallbackAtIndex (uint32_t idx)
{
    UnwindAssemblyProfilerInstance instance;
    if (AccessUnwindAssemblyProfilerInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

UnwindAssemblyProfilerCreateInstance
PluginManager::GetUnwindAssemblyProfilerCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        UnwindAssemblyProfilerInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessUnwindAssemblyProfilerInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}

#pragma mark ArchDefaultUnwindPlan

struct ArchDefaultUnwindPlanInstance
{
    ArchDefaultUnwindPlanInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ArchDefaultUnwindPlanCreateInstance create_callback;
};

typedef std::vector<ArchDefaultUnwindPlanInstance> ArchDefaultUnwindPlanInstances;

static bool
AccessArchDefaultUnwindPlanInstances (PluginAction action, ArchDefaultUnwindPlanInstance &instance, uint32_t index)
{
    static ArchDefaultUnwindPlanInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            ArchDefaultUnwindPlanInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ArchDefaultUnwindPlanCreateInstance create_callback
)
{
    if (create_callback)
    {
        ArchDefaultUnwindPlanInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessArchDefaultUnwindPlanInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ArchDefaultUnwindPlanCreateInstance create_callback)
{
    if (create_callback)
    {
        ArchDefaultUnwindPlanInstance instance;
        instance.create_callback = create_callback;
        return AccessArchDefaultUnwindPlanInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ArchDefaultUnwindPlanCreateInstance
PluginManager::GetArchDefaultUnwindPlanCreateCallbackAtIndex (uint32_t idx)
{
    ArchDefaultUnwindPlanInstance instance;
    if (AccessArchDefaultUnwindPlanInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

ArchDefaultUnwindPlanCreateInstance
PluginManager::GetArchDefaultUnwindPlanCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ArchDefaultUnwindPlanInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessArchDefaultUnwindPlanInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}

#pragma mark ArchVolatileRegs

struct ArchVolatileRegsInstance
{
    ArchVolatileRegsInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ArchVolatileRegsCreateInstance create_callback;
};

typedef std::vector<ArchVolatileRegsInstance> ArchVolatileRegsInstances;

static bool
AccessArchVolatileRegsInstances (PluginAction action, ArchVolatileRegsInstance &instance, uint32_t index)
{
    static ArchVolatileRegsInstances g_plugin_instances;

    switch (action)
    {
    case ePluginRegisterInstance:
        if (instance.create_callback)
        {
            g_plugin_instances.push_back (instance);
            return true;
        }
        break;

    case ePluginUnregisterInstance:
        if (instance.create_callback)
        {
            ArchVolatileRegsInstances::iterator pos, end = g_plugin_instances.end();
            for (pos = g_plugin_instances.begin(); pos != end; ++ pos)
            {
                if (pos->create_callback == instance.create_callback)
                {
                    g_plugin_instances.erase(pos);
                    return true;
                }
            }
        }
        break;

    case ePluginGetInstanceAtIndex:
        if (index < g_plugin_instances.size())
        {
            instance = g_plugin_instances[index];
            return true;
        }
        break;

    default:
        break;
    }
    return false;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ArchVolatileRegsCreateInstance create_callback
)
{
    if (create_callback)
    {
        ArchVolatileRegsInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        return AccessArchVolatileRegsInstances (ePluginRegisterInstance, instance, 0);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ArchVolatileRegsCreateInstance create_callback)
{
    if (create_callback)
    {
        ArchVolatileRegsInstance instance;
        instance.create_callback = create_callback;
        return AccessArchVolatileRegsInstances (ePluginUnregisterInstance, instance, 0);
    }
    return false;
}

ArchVolatileRegsCreateInstance
PluginManager::GetArchVolatileRegsCreateCallbackAtIndex (uint32_t idx)
{
    ArchVolatileRegsInstance instance;
    if (AccessArchVolatileRegsInstances (ePluginGetInstanceAtIndex, instance, idx))
        return instance.create_callback;
    return NULL;
}

ArchVolatileRegsCreateInstance
PluginManager::GetArchVolatileRegsCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        ArchVolatileRegsInstance instance;
        std::string ss_name(name);
        for (uint32_t idx = 0; AccessArchVolatileRegsInstances (ePluginGetInstanceAtIndex, instance, idx); ++idx)
        {
            if (instance.name == ss_name)
                return instance.create_callback;
        }
    }
    return NULL;
}


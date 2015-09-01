//===-- Language.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <map>
#include <mutex>

#include "lldb/Target/Language.h"

#include "lldb/Host/Mutex.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

typedef std::unique_ptr<Language> LanguageUP;
typedef std::map<lldb::LanguageType, LanguageUP> LanguagesMap;

static LanguagesMap&
GetLanguagesMap ()
{
    static LanguagesMap *g_map = nullptr;
    static std::once_flag g_initialize;
    
    std::call_once(g_initialize, [] {
        g_map = new LanguagesMap(); // NOTE: INTENTIONAL LEAK due to global destructor chain
    });
    
    return *g_map;
}
static Mutex&
GetLanguagesMutex ()
{
    static Mutex *g_mutex = nullptr;
    static std::once_flag g_initialize;
    
    std::call_once(g_initialize, [] {
        g_mutex = new Mutex(); // NOTE: INTENTIONAL LEAK due to global destructor chain
    });
    
    return *g_mutex;
}

Language*
Language::FindPlugin (lldb::LanguageType language)
{
    Mutex::Locker locker(GetLanguagesMutex());
    LanguagesMap& map(GetLanguagesMap());
    auto iter = map.find(language), end = map.end();
    if (iter != end)
        return iter->second.get();
    
    Language *language_ptr = nullptr;
    LanguageCreateInstance create_callback;
    
    for (uint32_t idx = 0;
         (create_callback = PluginManager::GetLanguageCreateCallbackAtIndex(idx)) != nullptr;
         ++idx)
    {
        language_ptr = create_callback(language);
        
        if (language_ptr)
        {
            map[language] = std::unique_ptr<Language>(language_ptr);
            return language_ptr;
        }
    }
    
    return nullptr;
}

void
Language::ForEach (std::function<bool(Language*)> callback)
{
    Mutex::Locker locker(GetLanguagesMutex());
    LanguagesMap& map(GetLanguagesMap());
    for (const auto& entry : map)
    {
        if (!callback(entry.second.get()))
            break;
    }
}

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
Language::Language()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Language::~Language()
{
}

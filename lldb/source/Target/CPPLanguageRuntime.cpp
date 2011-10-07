//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CPPLanguageRuntime::~CPPLanguageRuntime()
{
}

CPPLanguageRuntime::CPPLanguageRuntime (Process *process) :
    LanguageRuntime (process)
{

}

bool
CPPLanguageRuntime::GetObjectDescription (Stream &str, ValueObject &object)
{
    // C++ has no generic way to do this.
    return false;
}

bool
CPPLanguageRuntime::GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope)
{
    // C++ has no generic way to do this.
    return false;
}

bool
CPPLanguageRuntime::IsCPPMangledName (const char *name)
{
    // FIXME, we should really run through all the known C++ Language plugins and ask each one if
    // this is a C++ mangled name, but we can put that off till there is actually more than one
    // we care about.
    
    if (name && name[0] == '_' && name[1] == 'Z')
        return true;
    else
        return false;
}

bool
CPPLanguageRuntime::StripNamespacesFromVariableName (const char *name, const char *&base_name_start, const char *&base_name_end)
{
  if (base_name_end == NULL)
    base_name_end = name + strlen (name);
    
  const char *last_colon = NULL;
  for (const char *ptr = base_name_end; ptr != name; ptr--)
    {
      if (*ptr == ':')
        {
          last_colon = ptr;
          break;
        }
    }

  if (last_colon == NULL)
    {
      base_name_start = name;
      return true;
    }

  // Can't have a C++ name that begins with a single ':', nor contains an internal single ':'
  if (last_colon == name)
    return false;
  else if (last_colon[-1] != ':')
    return false;
  else
    {
      // FIXME: should check if there is
      base_name_start = last_colon + 1;
      return true;
    }
}
bool
CPPLanguageRuntime::IsPossibleCPPCall (const char *name, const char *&base_name_start, const char *&base_name_end)
{
    if (!name)
      return false;
    // For now, I really can't handle taking template names apart, so if you
    // have < or > I'll say "could be CPP but leave the base_name empty which
    // means I couldn't figure out what to use for that.
    // FIXME: Do I need to do more sanity checking here?

    if (strchr(name, '>') != NULL || strchr (name, '>') != NULL)
      return true;

    size_t name_len = strlen (name);

    if (name[name_len - 1] == ')')
    {
        // We've got arguments.
        base_name_end = strchr (name, '(');
        if (base_name_end == NULL)
          return false;

        // FIXME: should check that this parenthesis isn't a template specialized
        // on a function type or something gross like that...
    }
    else
        base_name_end = name + strlen (name);

    return StripNamespacesFromVariableName (name, base_name_start, base_name_end);
}

//===-- ObjCLanguage.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>

#include "ObjCLanguage.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/Threading.h"

#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

#include "CF.h"
#include "Cocoa.h"
#include "CoreMedia.h"
#include "NSDictionary.h"
#include "NSSet.h"
#include "NSString.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(ObjCLanguage)

void ObjCLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Objective-C Language",
                                CreateInstance);
}

void ObjCLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString ObjCLanguage::GetPluginNameStatic() {
  static ConstString g_name("objc");
  return g_name;
}

// PluginInterface protocol

lldb_private::ConstString ObjCLanguage::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ObjCLanguage::GetPluginVersion() { return 1; }

// Static Functions

Language *ObjCLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeObjC:
    return new ObjCLanguage();
  default:
    return nullptr;
  }
}

void ObjCLanguage::MethodName::Clear() {
  m_full.Clear();
  m_class.Clear();
  m_category.Clear();
  m_selector.Clear();
  m_type = eTypeUnspecified;
  m_category_is_valid = false;
}

bool ObjCLanguage::MethodName::SetName(llvm::StringRef name, bool strict) {
  Clear();
  if (name.empty())
    return IsValid(strict);

  // If "strict" is true. then the method must be specified with a '+' or '-'
  // at the beginning. If "strict" is false, then the '+' or '-' can be omitted
  bool valid_prefix = false;

  if (name.size() > 1 && (name[0] == '+' || name[0] == '-')) {
    valid_prefix = name[1] == '[';
    if (name[0] == '+')
      m_type = eTypeClassMethod;
    else
      m_type = eTypeInstanceMethod;
  } else if (!strict) {
    // "strict" is false, the name just needs to start with '['
    valid_prefix = name[0] == '[';
  }

  if (valid_prefix) {
    int name_len = name.size();
    // Objective-C methods must have at least:
    //      "-[" or "+[" prefix
    //      One character for a class name
    //      One character for the space between the class name
    //      One character for the method name
    //      "]" suffix
    if (name_len >= (5 + (strict ? 1 : 0)) && name.back() == ']') {
      m_full.SetString(name);
    }
  }
  return IsValid(strict);
}

bool ObjCLanguage::MethodName::SetName(const char *name, bool strict) {
  return SetName(llvm::StringRef(name), strict);
}

ConstString ObjCLanguage::MethodName::GetClassName() {
  if (!m_class) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *paren_pos = strchr(class_start, '(');
      if (paren_pos) {
        m_class.SetCStringWithLength(class_start, paren_pos - class_start);
      } else {
        // No '(' was found in the full name, we can definitively say that our
        // category was valid (and empty).
        m_category_is_valid = true;
        const char *space_pos = strchr(full, ' ');
        if (space_pos) {
          m_class.SetCStringWithLength(class_start, space_pos - class_start);
          if (!m_class_category) {
            // No category in name, so we can also fill in the m_class_category
            m_class_category = m_class;
          }
        }
      }
    }
  }
  return m_class;
}

ConstString ObjCLanguage::MethodName::GetClassNameWithCategory() {
  if (!m_class_category) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *space_pos = strchr(full, ' ');
      if (space_pos) {
        m_class_category.SetCStringWithLength(class_start,
                                              space_pos - class_start);
        // If m_class hasn't been filled in and the class with category doesn't
        // contain a '(', then we can also fill in the m_class
        if (!m_class && strchr(m_class_category.GetCString(), '(') == nullptr) {
          m_class = m_class_category;
          // No '(' was found in the full name, we can definitively say that
          // our category was valid (and empty).
          m_category_is_valid = true;
        }
      }
    }
  }
  return m_class_category;
}

ConstString ObjCLanguage::MethodName::GetSelector() {
  if (!m_selector) {
    if (IsValid(false)) {
      const char *full = m_full.GetCString();
      const char *space_pos = strchr(full, ' ');
      if (space_pos) {
        ++space_pos; // skip the space
        m_selector.SetCStringWithLength(space_pos, m_full.GetLength() -
                                                       (space_pos - full) - 1);
      }
    }
  }
  return m_selector;
}

ConstString ObjCLanguage::MethodName::GetCategory() {
  if (!m_category_is_valid && !m_category) {
    if (IsValid(false)) {
      m_category_is_valid = true;
      const char *full = m_full.GetCString();
      const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
      const char *open_paren_pos = strchr(class_start, '(');
      if (open_paren_pos) {
        ++open_paren_pos; // Skip the open paren
        const char *close_paren_pos = strchr(open_paren_pos, ')');
        if (close_paren_pos)
          m_category.SetCStringWithLength(open_paren_pos,
                                          close_paren_pos - open_paren_pos);
      }
    }
  }
  return m_category;
}

ConstString ObjCLanguage::MethodName::GetFullNameWithoutCategory(
    bool empty_if_no_category) {
  if (IsValid(false)) {
    if (HasCategory()) {
      StreamString strm;
      if (m_type == eTypeClassMethod)
        strm.PutChar('+');
      else if (m_type == eTypeInstanceMethod)
        strm.PutChar('-');
      strm.Printf("[%s %s]", GetClassName().GetCString(),
                  GetSelector().GetCString());
      return ConstString(strm.GetString());
    }

    if (!empty_if_no_category) {
      // Just return the full name since it doesn't have a category
      return GetFullName();
    }
  }
  return ConstString();
}

std::vector<Language::MethodNameVariant>
ObjCLanguage::GetMethodNameVariants(ConstString method_name) const {
  std::vector<Language::MethodNameVariant> variant_names;
  ObjCLanguage::MethodName objc_method(method_name.GetCString(), false);
  if (!objc_method.IsValid(false)) {
    return variant_names;
  }

  variant_names.emplace_back(objc_method.GetSelector(),
                             lldb::eFunctionNameTypeSelector);

  const bool is_class_method =
      objc_method.GetType() == MethodName::eTypeClassMethod;
  const bool is_instance_method =
      objc_method.GetType() == MethodName::eTypeInstanceMethod;
  ConstString name_sans_category =
      objc_method.GetFullNameWithoutCategory(/*empty_if_no_category*/ true);

  if (is_class_method || is_instance_method) {
    if (name_sans_category)
      variant_names.emplace_back(name_sans_category,
                                 lldb::eFunctionNameTypeFull);
  } else {
    StreamString strm;

    strm.Printf("+%s", objc_method.GetFullName().GetCString());
    variant_names.emplace_back(ConstString(strm.GetString()),
                               lldb::eFunctionNameTypeFull);
    strm.Clear();

    strm.Printf("-%s", objc_method.GetFullName().GetCString());
    variant_names.emplace_back(ConstString(strm.GetString()),
                               lldb::eFunctionNameTypeFull);
    strm.Clear();

    if (name_sans_category) {
      strm.Printf("+%s", name_sans_category.GetCString());
      variant_names.emplace_back(ConstString(strm.GetString()),
                                 lldb::eFunctionNameTypeFull);
      strm.Clear();

      strm.Printf("-%s", name_sans_category.GetCString());
      variant_names.emplace_back(ConstString(strm.GetString()),
                                 lldb::eFunctionNameTypeFull);
    }
  }

  return variant_names;
}

bool ObjCLanguage::SymbolNameFitsToLanguage(Mangled mangled) const {
  ConstString demangled_name = mangled.GetDemangledName();
  if (!demangled_name)
    return false;
  return ObjCLanguage::IsPossibleObjCMethodName(demangled_name.GetCString());
}

static void LoadObjCFormatters(TypeCategoryImplSP objc_category_sp) {
  if (!objc_category_sp)
    return;

  TypeSummaryImpl::Flags objc_flags;
  objc_flags.SetCascades(false)
      .SetSkipPointers(true)
      .SetSkipReferences(true)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  lldb::TypeSummaryImplSP ObjC_BOOL_summary(new CXXFunctionSummaryFormat(
      objc_flags, lldb_private::formatters::ObjCBOOLSummaryProvider, ""));
  objc_category_sp->GetTypeSummariesContainer()->Add(ConstString("BOOL"),
                                                     ObjC_BOOL_summary);
  objc_category_sp->GetTypeSummariesContainer()->Add(ConstString("BOOL &"),
                                                     ObjC_BOOL_summary);
  objc_category_sp->GetTypeSummariesContainer()->Add(ConstString("BOOL *"),
                                                     ObjC_BOOL_summary);

  // we need to skip pointers here since we are special casing a SEL* when
  // retrieving its value
  objc_flags.SetSkipPointers(true);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<false>,
                "SEL summary provider", ConstString("SEL"), objc_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>,
      "SEL summary provider", ConstString("struct objc_selector"), objc_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>,
      "SEL summary provider", ConstString("objc_selector"), objc_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<true>,
      "SEL summary provider", ConstString("objc_selector *"), objc_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCSELSummaryProvider<true>,
                "SEL summary provider", ConstString("SEL *"), objc_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::ObjCClassSummaryProvider,
                "Class summary provider", ConstString("Class"), objc_flags);

  SyntheticChildren::Flags class_synth_flags;
  class_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(
      false);

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::ObjCClassSyntheticFrontEndCreator,
                  "Class synthetic children", ConstString("Class"),
                  class_synth_flags);

  objc_flags.SetSkipPointers(false);
  objc_flags.SetCascades(true);
  objc_flags.SetSkipReferences(false);

  AddStringSummary(objc_category_sp, "${var.__FuncPtr%A}",
                   ConstString("__block_literal_generic"), objc_flags);

  AddStringSummary(objc_category_sp, "${var.years} years, ${var.months} "
                                     "months, ${var.days} days, ${var.hours} "
                                     "hours, ${var.minutes} minutes "
                                     "${var.seconds} seconds",
                   ConstString("CFGregorianUnits"), objc_flags);
  AddStringSummary(objc_category_sp,
                   "location=${var.location} length=${var.length}",
                   ConstString("CFRange"), objc_flags);

  AddStringSummary(objc_category_sp,
                   "location=${var.location}, length=${var.length}",
                   ConstString("NSRange"), objc_flags);
  AddStringSummary(objc_category_sp, "(${var.origin}, ${var.size}), ...",
                   ConstString("NSRectArray"), objc_flags);

  AddOneLineSummary(objc_category_sp, ConstString("NSPoint"), objc_flags);
  AddOneLineSummary(objc_category_sp, ConstString("NSSize"), objc_flags);
  AddOneLineSummary(objc_category_sp, ConstString("NSRect"), objc_flags);

  AddOneLineSummary(objc_category_sp, ConstString("CGSize"), objc_flags);
  AddOneLineSummary(objc_category_sp, ConstString("CGPoint"), objc_flags);
  AddOneLineSummary(objc_category_sp, ConstString("CGRect"), objc_flags);

  AddStringSummary(objc_category_sp,
                   "red=${var.red} green=${var.green} blue=${var.blue}",
                   ConstString("RGBColor"), objc_flags);
  AddStringSummary(
      objc_category_sp,
      "(t=${var.top}, l=${var.left}, b=${var.bottom}, r=${var.right})",
      ConstString("Rect"), objc_flags);
  AddStringSummary(objc_category_sp, "{(v=${var.v}, h=${var.h})}",
                   ConstString("Point"), objc_flags);
  AddStringSummary(objc_category_sp,
                   "${var.month}/${var.day}/${var.year}  ${var.hour} "
                   ":${var.minute} :${var.second} dayOfWeek:${var.dayOfWeek}",
                   ConstString("DateTimeRect *"), objc_flags);
  AddStringSummary(objc_category_sp, "${var.ld.month}/${var.ld.day}/"
                                     "${var.ld.year} ${var.ld.hour} "
                                     ":${var.ld.minute} :${var.ld.second} "
                                     "dayOfWeek:${var.ld.dayOfWeek}",
                   ConstString("LongDateRect"), objc_flags);
  AddStringSummary(objc_category_sp, "(x=${var.x}, y=${var.y})",
                   ConstString("HIPoint"), objc_flags);
  AddStringSummary(objc_category_sp, "origin=${var.origin} size=${var.size}",
                   ConstString("HIRect"), objc_flags);

  TypeSummaryImpl::Flags appkit_flags;
  appkit_flags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(false)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  appkit_flags.SetDontShowChildren(false);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("NSArray"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("NSConstantArray"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("NSMutableArray"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("__NSArrayI"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("__NSArray0"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider",
                ConstString("__NSSingleObjectArrayI"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("__NSArrayM"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("__NSCFArray"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("_NSCallStackArray"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSArraySummaryProvider,
      "NSArray summary provider", ConstString("CFArrayRef"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSArraySummaryProvider,
                "NSArray summary provider", ConstString("CFMutableArrayRef"),
                appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", ConstString("NSDictionary"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider",
                ConstString("NSConstantDictionary"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider",
                ConstString("NSMutableDictionary"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider",
                ConstString("__NSCFDictionary"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", ConstString("__NSDictionaryI"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider",
                ConstString("__NSSingleEntryDictionaryI"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<false>,
                "NSDictionary summary provider", ConstString("__NSDictionaryM"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider", ConstString("CFDictionaryRef"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider", ConstString("__CFDictionary"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDictionarySummaryProvider<true>,
                "NSDictionary summary provider",
                ConstString("CFMutableDictionaryRef"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "NSSet summary", ConstString("NSSet"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "NSMutableSet summary", ConstString("NSMutableSet"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<true>,
                "CFSetRef summary", ConstString("CFSetRef"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<true>,
      "CFMutableSetRef summary", ConstString("CFMutableSetRef"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSCFSet summary", ConstString("__NSCFSet"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__CFSet summary", ConstString("__CFSet"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSSetI summary", ConstString("__NSSetI"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSSetSummaryProvider<false>,
                "__NSSetM summary", ConstString("__NSSetM"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "NSCountedSet summary", ConstString("NSCountedSet"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "NSMutableSet summary", ConstString("NSMutableSet"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "NSOrderedSet summary", ConstString("NSOrderedSet"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "__NSOrderedSetI summary", ConstString("__NSOrderedSetI"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>,
      "__NSOrderedSetM summary", ConstString("__NSOrderedSetM"), appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSError_SummaryProvider,
      "NSError summary provider", ConstString("NSError"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSException_SummaryProvider,
      "NSException summary provider", ConstString("NSException"), appkit_flags);

  // AddSummary(appkit_category_sp, "${var.key%@} -> ${var.value%@}",
  // ConstString("$_lldb_typegen_nspair"), appkit_flags);

  appkit_flags.SetDontShowChildren(true);

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("__NSArrayM"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("__NSArrayI"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("__NSArray0"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children",
                  ConstString("__NSSingleObjectArrayI"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("NSArray"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("NSConstantArray"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("NSMutableArray"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("__NSCFArray"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("_NSCallStackArray"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children",
                  ConstString("CFMutableArrayRef"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSArraySyntheticFrontEndCreator,
                  "NSArray synthetic children", ConstString("CFArrayRef"),
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("__NSDictionaryM"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("NSConstantDictionary"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("__NSDictionaryI"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children",
      ConstString("__NSSingleEntryDictionaryI"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("__NSCFDictionary"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("NSDictionary"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("NSMutableDictionary"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("CFDictionaryRef"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("CFMutableDictionaryRef"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp,
      lldb_private::formatters::NSDictionarySyntheticFrontEndCreator,
      "NSDictionary synthetic children", ConstString("__CFDictionary"),
      ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSErrorSyntheticFrontEndCreator,
                  "NSError synthetic children", ConstString("NSError"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSExceptionSyntheticFrontEndCreator,
                  "NSException synthetic children", ConstString("NSException"),
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "NSSet synthetic children", ConstString("NSSet"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSSetI synthetic children", ConstString("__NSSetI"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSSetM synthetic children", ConstString("__NSSetM"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__NSCFSet synthetic children", ConstString("__NSCFSet"),
                  ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "CFSetRef synthetic children", ConstString("CFSetRef"),
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(
      objc_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator,
      "NSMutableSet synthetic children", ConstString("NSMutableSet"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator,
      "NSOrderedSet synthetic children", ConstString("NSOrderedSet"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator,
      "__NSOrderedSetI synthetic children", ConstString("__NSOrderedSetI"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(
      objc_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator,
      "__NSOrderedSetM synthetic children", ConstString("__NSOrderedSetM"),
      ScriptedSyntheticChildren::Flags());
  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSSetSyntheticFrontEndCreator,
                  "__CFSet synthetic children", ConstString("__CFSet"),
                  ScriptedSyntheticChildren::Flags());

  AddCXXSynthetic(objc_category_sp,
                  lldb_private::formatters::NSIndexPathSyntheticFrontEndCreator,
                  "NSIndexPath synthetic children", ConstString("NSIndexPath"),
                  ScriptedSyntheticChildren::Flags());

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBagSummaryProvider,
      "CFBag summary provider", ConstString("CFBagRef"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", ConstString("__CFBag"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBagSummaryProvider,
                "CFBag summary provider", ConstString("const struct __CFBag"),
                appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::CFBagSummaryProvider,
      "CFBag summary provider", ConstString("CFMutableBagRef"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBinaryHeapSummaryProvider,
                "CFBinaryHeap summary provider", ConstString("CFBinaryHeapRef"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBinaryHeapSummaryProvider,
                "CFBinaryHeap summary provider", ConstString("__CFBinaryHeap"),
                appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("NSString"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("CFStringRef"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("__CFString"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", ConstString("CFMutableStringRef"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", ConstString("NSMutableString"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider",
                ConstString("__NSCFConstantString"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("__NSCFString"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider", ConstString("NSCFConstantString"),
                appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("NSCFString"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSStringSummaryProvider,
      "NSString summary provider", ConstString("NSPathStore2"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSStringSummaryProvider,
                "NSString summary provider",
                ConstString("NSTaggedPointerString"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSAttributedStringSummaryProvider,
                "NSAttributedString summary provider",
                ConstString("NSAttributedString"), appkit_flags);
  AddCXXSummary(
      objc_category_sp,
      lldb_private::formatters::NSMutableAttributedStringSummaryProvider,
      "NSMutableAttributedString summary provider",
      ConstString("NSMutableAttributedString"), appkit_flags);
  AddCXXSummary(
      objc_category_sp,
      lldb_private::formatters::NSMutableAttributedStringSummaryProvider,
      "NSMutableAttributedString summary provider",
      ConstString("NSConcreteMutableAttributedString"), appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSBundleSummaryProvider,
      "NSBundle summary provider", ConstString("NSBundle"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", ConstString("NSData"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>,
      "NSData summary provider", ConstString("_NSInlineData"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>,
      "NSData summary provider", ConstString("NSConcreteData"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDataSummaryProvider<false>,
                "NSData summary provider", ConstString("NSConcreteMutableData"),
                appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>,
      "NSData summary provider", ConstString("NSMutableData"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>,
      "NSData summary provider", ConstString("__NSCFData"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>,
      "NSData summary provider", ConstString("CFDataRef"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>,
      "NSData summary provider", ConstString("CFMutableDataRef"), appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSMachPortSummaryProvider,
      "NSMachPort summary provider", ConstString("NSMachPort"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNotificationSummaryProvider,
                "NSNotification summary provider",
                ConstString("NSNotification"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNotificationSummaryProvider,
                "NSNotification summary provider",
                ConstString("NSConcreteNotification"), appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", ConstString("NSNumber"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider",
                ConstString("NSConstantIntegerNumber"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider",
                ConstString("NSConstantDoubleNumber"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSNumber summary provider",
                ConstString("NSConstantFloatNumber"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "CFNumberRef summary provider", ConstString("CFNumberRef"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", ConstString("__NSCFBoolean"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", ConstString("__NSCFNumber"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", ConstString("NSCFBoolean"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSNumberSummaryProvider,
      "NSNumber summary provider", ConstString("NSCFNumber"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSNumberSummaryProvider,
                "NSDecimalNumber summary provider",
                ConstString("NSDecimalNumber"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSURLSummaryProvider,
                "NSURL summary provider", ConstString("NSURL"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSURLSummaryProvider,
      "NSURL summary provider", ConstString("CFURLRef"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSDateSummaryProvider,
                "NSDate summary provider", ConstString("NSDate"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDateSummaryProvider,
      "NSDate summary provider", ConstString("__NSDate"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDateSummaryProvider,
      "NSDate summary provider", ConstString("__NSTaggedDate"), appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSDateSummaryProvider,
      "NSDate summary provider", ConstString("NSCalendarDate"), appkit_flags);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSTimeZoneSummaryProvider,
      "NSTimeZone summary provider", ConstString("NSTimeZone"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSTimeZoneSummaryProvider,
                "NSTimeZone summary provider", ConstString("CFTimeZoneRef"),
                appkit_flags);
  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSTimeZoneSummaryProvider,
      "NSTimeZone summary provider", ConstString("__NSTimeZone"), appkit_flags);

  // CFAbsoluteTime is actually a double rather than a pointer to an object we
  // do not care about the numeric value, since it is probably meaningless to
  // users
  appkit_flags.SetDontShowValue(true);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFAbsoluteTimeSummaryProvider,
                "CFAbsoluteTime summary provider",
                ConstString("CFAbsoluteTime"), appkit_flags);
  appkit_flags.SetDontShowValue(false);

  AddCXXSummary(
      objc_category_sp, lldb_private::formatters::NSIndexSetSummaryProvider,
      "NSIndexSet summary provider", ConstString("NSIndexSet"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::NSIndexSetSummaryProvider,
                "NSIndexSet summary provider", ConstString("NSMutableIndexSet"),
                appkit_flags);

  AddStringSummary(objc_category_sp,
                   "@\"${var.month%d}/${var.day%d}/${var.year%d} "
                   "${var.hour%d}:${var.minute%d}:${var.second}\"",
                   ConstString("CFGregorianDate"), appkit_flags);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider", ConstString("CFBitVectorRef"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider",
                ConstString("CFMutableBitVectorRef"), appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider", ConstString("__CFBitVector"),
                appkit_flags);
  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CFBitVectorSummaryProvider,
                "CFBitVector summary provider",
                ConstString("__CFMutableBitVector"), appkit_flags);
}

static void LoadCoreMediaFormatters(TypeCategoryImplSP objc_category_sp) {
  if (!objc_category_sp)
    return;

  TypeSummaryImpl::Flags cm_flags;
  cm_flags.SetCascades(true)
      .SetDontShowChildren(false)
      .SetDontShowValue(false)
      .SetHideItemNames(false)
      .SetShowMembersOneLiner(false)
      .SetSkipPointers(false)
      .SetSkipReferences(false);

  AddCXXSummary(objc_category_sp,
                lldb_private::formatters::CMTimeSummaryProvider,
                "CMTime summary provider", ConstString("CMTime"), cm_flags);
}

lldb::TypeCategoryImplSP ObjCLanguage::GetFormatters() {
  static llvm::once_flag g_initialize;
  static TypeCategoryImplSP g_category;

  llvm::call_once(g_initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(GetPluginName(), g_category);
    if (g_category) {
      LoadCoreMediaFormatters(g_category);
      LoadObjCFormatters(g_category);
    }
  });
  return g_category;
}

std::vector<ConstString>
ObjCLanguage::GetPossibleFormattersMatches(ValueObject &valobj,
                                           lldb::DynamicValueType use_dynamic) {
  std::vector<ConstString> result;

  if (use_dynamic == lldb::eNoDynamicValues)
    return result;

  CompilerType compiler_type(valobj.GetCompilerType());

  const bool check_cpp = false;
  const bool check_objc = true;
  bool canBeObjCDynamic =
      compiler_type.IsPossibleDynamicType(nullptr, check_cpp, check_objc);

  if (canBeObjCDynamic && ClangUtil::IsClangType(compiler_type)) {
    do {
      lldb::ProcessSP process_sp = valobj.GetProcessSP();
      if (!process_sp)
        break;
      ObjCLanguageRuntime *runtime = ObjCLanguageRuntime::Get(*process_sp);
      if (runtime == nullptr)
        break;
      ObjCLanguageRuntime::ClassDescriptorSP objc_class_sp(
          runtime->GetClassDescriptor(valobj));
      if (!objc_class_sp)
        break;
      if (ConstString name = objc_class_sp->GetClassName())
        result.push_back(name);
    } while (false);
  }

  return result;
}

std::unique_ptr<Language::TypeScavenger> ObjCLanguage::GetTypeScavenger() {
  class ObjCScavengerResult : public Language::TypeScavenger::Result {
  public:
    ObjCScavengerResult(CompilerType type)
        : Language::TypeScavenger::Result(), m_compiler_type(type) {}

    bool IsValid() override { return m_compiler_type.IsValid(); }

    bool DumpToStream(Stream &stream, bool print_help_if_available) override {
      if (IsValid()) {
        m_compiler_type.DumpTypeDescription(&stream);
        stream.EOL();
        return true;
      }
      return false;
    }

  private:
    CompilerType m_compiler_type;
  };

  class ObjCRuntimeScavenger : public Language::TypeScavenger {
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      bool result = false;

      if (auto *process = exe_scope->CalculateProcess().get()) {
        if (auto *objc_runtime = ObjCLanguageRuntime::Get(*process)) {
          if (auto *decl_vendor = objc_runtime->GetDeclVendor()) {
            ConstString name(key);
            for (const CompilerType &type :
                 decl_vendor->FindTypes(name, /*max_matches*/ UINT32_MAX)) {
              result = true;
              std::unique_ptr<Language::TypeScavenger::Result> result(
                  new ObjCScavengerResult(type));
              results.insert(std::move(result));
            }
          }
        }
      }

      return result;
    }

    friend class lldb_private::ObjCLanguage;
  };

  class ObjCModulesScavenger : public Language::TypeScavenger {
  protected:
    bool Find_Impl(ExecutionContextScope *exe_scope, const char *key,
                   ResultSet &results) override {
      bool result = false;

      if (auto *target = exe_scope->CalculateTarget().get()) {
        auto *persistent_vars = llvm::cast<ClangPersistentVariables>(
            target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeC));
        if (std::shared_ptr<ClangModulesDeclVendor> clang_modules_decl_vendor =
                persistent_vars->GetClangModulesDeclVendor()) {
          ConstString key_cs(key);
          auto types = clang_modules_decl_vendor->FindTypes(
              key_cs, /*max_matches*/ UINT32_MAX);
          if (!types.empty()) {
            result = true;
            std::unique_ptr<Language::TypeScavenger::Result> result(
                new ObjCScavengerResult(types.front()));
            results.insert(std::move(result));
          }
        }
      }

      return result;
    }

    friend class lldb_private::ObjCLanguage;
  };
  
  class ObjCDebugInfoScavenger : public Language::ImageListTypeScavenger {
  public:
    CompilerType AdjustForInclusion(CompilerType &candidate) override {
      LanguageType lang_type(candidate.GetMinimumLanguage());
      if (!Language::LanguageIsObjC(lang_type))
        return CompilerType();
      if (candidate.IsTypedefType())
        return candidate.GetTypedefedType();
      return candidate;
    }
  };

  return std::unique_ptr<TypeScavenger>(
      new Language::EitherTypeScavenger<ObjCModulesScavenger,
                                        ObjCRuntimeScavenger,
                                        ObjCDebugInfoScavenger>());
}

bool ObjCLanguage::GetFormatterPrefixSuffix(ValueObject &valobj,
                                            ConstString type_hint,
                                            std::string &prefix,
                                            std::string &suffix) {
  static ConstString g_CFBag("CFBag");
  static ConstString g_CFBinaryHeap("CFBinaryHeap");

  static ConstString g_NSNumberChar("NSNumber:char");
  static ConstString g_NSNumberShort("NSNumber:short");
  static ConstString g_NSNumberInt("NSNumber:int");
  static ConstString g_NSNumberLong("NSNumber:long");
  static ConstString g_NSNumberInt128("NSNumber:int128_t");
  static ConstString g_NSNumberFloat("NSNumber:float");
  static ConstString g_NSNumberDouble("NSNumber:double");

  static ConstString g_NSData("NSData");
  static ConstString g_NSArray("NSArray");
  static ConstString g_NSString("NSString");
  static ConstString g_NSStringStar("NSString*");

  if (type_hint.IsEmpty())
    return false;

  prefix.clear();
  suffix.clear();

  if (type_hint == g_CFBag || type_hint == g_CFBinaryHeap) {
    prefix = "@";
    return true;
  }

  if (type_hint == g_NSNumberChar) {
    prefix = "(char)";
    return true;
  }
  if (type_hint == g_NSNumberShort) {
    prefix = "(short)";
    return true;
  }
  if (type_hint == g_NSNumberInt) {
    prefix = "(int)";
    return true;
  }
  if (type_hint == g_NSNumberLong) {
    prefix = "(long)";
    return true;
  }
  if (type_hint == g_NSNumberInt128) {
    prefix = "(int128_t)";
    return true;
  }
  if (type_hint == g_NSNumberFloat) {
    prefix = "(float)";
    return true;
  }
  if (type_hint == g_NSNumberDouble) {
    prefix = "(double)";
    return true;
  }

  if (type_hint == g_NSData || type_hint == g_NSArray) {
    prefix = "@\"";
    suffix = "\"";
    return true;
  }

  if (type_hint == g_NSString || type_hint == g_NSStringStar) {
    prefix = "@";
    return true;
  }

  return false;
}

bool ObjCLanguage::IsNilReference(ValueObject &valobj) {
  const uint32_t mask = eTypeIsObjC | eTypeIsPointer;
  bool isObjCpointer =
      (((valobj.GetCompilerType().GetTypeInfo(nullptr)) & mask) == mask);
  if (!isObjCpointer)
    return false;
  bool canReadValue = true;
  bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
  return canReadValue && isZero;
}

bool ObjCLanguage::IsSourceFile(llvm::StringRef file_path) const {
  const auto suffixes = {".h", ".m", ".M"};
  for (auto suffix : suffixes) {
    if (file_path.endswith_insensitive(suffix))
      return true;
  }
  return false;
}

//===-- TypeCategory.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_TYPECATEGORY_H
#define LLDB_DATAFORMATTERS_TYPECATEGORY_H

#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"

#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/FormattersContainer.h"

namespace lldb_private {

template <typename FormatterImpl> class FormatterContainerPair {
public:
  typedef FormattersContainer<FormatterImpl> ExactMatchContainer;
  typedef FormattersContainer<FormatterImpl> RegexMatchContainer;

  typedef TypeMatcher ExactMatchMap;
  typedef TypeMatcher RegexMatchMap;

  typedef typename ExactMatchContainer::ValueSP MapValueType;

  typedef typename ExactMatchContainer::SharedPointer ExactMatchContainerSP;
  typedef typename RegexMatchContainer::SharedPointer RegexMatchContainerSP;

  typedef
      typename ExactMatchContainer::ForEachCallback ExactMatchForEachCallback;
  typedef
      typename RegexMatchContainer::ForEachCallback RegexMatchForEachCallback;

  FormatterContainerPair(IFormatChangeListener *clist)
      : m_exact_sp(new ExactMatchContainer(clist)),
        m_regex_sp(new RegexMatchContainer(clist)) {}

  ~FormatterContainerPair() = default;

  ExactMatchContainerSP GetExactMatch() const { return m_exact_sp; }

  RegexMatchContainerSP GetRegexMatch() const { return m_regex_sp; }

  uint32_t GetCount() {
    return GetExactMatch()->GetCount() + GetRegexMatch()->GetCount();
  }

private:
  ExactMatchContainerSP m_exact_sp;
  RegexMatchContainerSP m_regex_sp;
};

class TypeCategoryImpl {
private:
  typedef FormatterContainerPair<TypeFormatImpl> FormatContainer;
  typedef FormatterContainerPair<TypeSummaryImpl> SummaryContainer;
  typedef FormatterContainerPair<TypeFilterImpl> FilterContainer;
  typedef FormatterContainerPair<SyntheticChildren> SynthContainer;

public:
  typedef uint16_t FormatCategoryItems;
  static const uint16_t ALL_ITEM_TYPES = UINT16_MAX;

  typedef FormatContainer::ExactMatchContainerSP FormatContainerSP;
  typedef FormatContainer::RegexMatchContainerSP RegexFormatContainerSP;

  typedef SummaryContainer::ExactMatchContainerSP SummaryContainerSP;
  typedef SummaryContainer::RegexMatchContainerSP RegexSummaryContainerSP;

  typedef FilterContainer::ExactMatchContainerSP FilterContainerSP;
  typedef FilterContainer::RegexMatchContainerSP RegexFilterContainerSP;

  typedef SynthContainer::ExactMatchContainerSP SynthContainerSP;
  typedef SynthContainer::RegexMatchContainerSP RegexSynthContainerSP;

  template <typename T> class ForEachCallbacks {
  public:
    ForEachCallbacks() = default;
    ~ForEachCallbacks() = default;

    template <typename U = TypeFormatImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetExact(FormatContainer::ExactMatchForEachCallback callback) {
      m_format_exact = std::move(callback);
      return *this;
    }
    template <typename U = TypeFormatImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetWithRegex(FormatContainer::RegexMatchForEachCallback callback) {
      m_format_regex = std::move(callback);
      return *this;
    }

    template <typename U = TypeSummaryImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetExact(SummaryContainer::ExactMatchForEachCallback callback) {
      m_summary_exact = std::move(callback);
      return *this;
    }
    template <typename U = TypeSummaryImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetWithRegex(SummaryContainer::RegexMatchForEachCallback callback) {
      m_summary_regex = std::move(callback);
      return *this;
    }

    template <typename U = TypeFilterImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetExact(FilterContainer::ExactMatchForEachCallback callback) {
      m_filter_exact = std::move(callback);
      return *this;
    }
    template <typename U = TypeFilterImpl>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetWithRegex(FilterContainer::RegexMatchForEachCallback callback) {
      m_filter_regex = std::move(callback);
      return *this;
    }

    template <typename U = SyntheticChildren>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetExact(SynthContainer::ExactMatchForEachCallback callback) {
      m_synth_exact = std::move(callback);
      return *this;
    }
    template <typename U = SyntheticChildren>
    typename std::enable_if<std::is_same<U, T>::value, ForEachCallbacks &>::type
    SetWithRegex(SynthContainer::RegexMatchForEachCallback callback) {
      m_synth_regex = std::move(callback);
      return *this;
    }

    FormatContainer::ExactMatchForEachCallback GetFormatExactCallback() const {
      return m_format_exact;
    }
    FormatContainer::RegexMatchForEachCallback GetFormatRegexCallback() const {
      return m_format_regex;
    }

    SummaryContainer::ExactMatchForEachCallback
    GetSummaryExactCallback() const {
      return m_summary_exact;
    }
    SummaryContainer::RegexMatchForEachCallback
    GetSummaryRegexCallback() const {
      return m_summary_regex;
    }

    FilterContainer::ExactMatchForEachCallback GetFilterExactCallback() const {
      return m_filter_exact;
    }
    FilterContainer::RegexMatchForEachCallback GetFilterRegexCallback() const {
      return m_filter_regex;
    }

    SynthContainer::ExactMatchForEachCallback GetSynthExactCallback() const {
      return m_synth_exact;
    }
    SynthContainer::RegexMatchForEachCallback GetSynthRegexCallback() const {
      return m_synth_regex;
    }

  private:
    FormatContainer::ExactMatchForEachCallback m_format_exact;
    FormatContainer::RegexMatchForEachCallback m_format_regex;

    SummaryContainer::ExactMatchForEachCallback m_summary_exact;
    SummaryContainer::RegexMatchForEachCallback m_summary_regex;

    FilterContainer::ExactMatchForEachCallback m_filter_exact;
    FilterContainer::RegexMatchForEachCallback m_filter_regex;

    SynthContainer::ExactMatchForEachCallback m_synth_exact;
    SynthContainer::RegexMatchForEachCallback m_synth_regex;
  };

  TypeCategoryImpl(IFormatChangeListener *clist, ConstString name);

  template <typename T> void ForEach(const ForEachCallbacks<T> &foreach) {
    GetTypeFormatsContainer()->ForEach(foreach.GetFormatExactCallback());
    GetRegexTypeFormatsContainer()->ForEach(foreach.GetFormatRegexCallback());

    GetTypeSummariesContainer()->ForEach(foreach.GetSummaryExactCallback());
    GetRegexTypeSummariesContainer()->ForEach(
        foreach.GetSummaryRegexCallback());

    GetTypeFiltersContainer()->ForEach(foreach.GetFilterExactCallback());
    GetRegexTypeFiltersContainer()->ForEach(foreach.GetFilterRegexCallback());

    GetTypeSyntheticsContainer()->ForEach(foreach.GetSynthExactCallback());
    GetRegexTypeSyntheticsContainer()->ForEach(foreach.GetSynthRegexCallback());
  }

  FormatContainerSP GetTypeFormatsContainer() {
    return m_format_cont.GetExactMatch();
  }

  RegexFormatContainerSP GetRegexTypeFormatsContainer() {
    return m_format_cont.GetRegexMatch();
  }

  FormatContainer &GetFormatContainer() { return m_format_cont; }

  SummaryContainerSP GetTypeSummariesContainer() {
    return m_summary_cont.GetExactMatch();
  }

  RegexSummaryContainerSP GetRegexTypeSummariesContainer() {
    return m_summary_cont.GetRegexMatch();
  }

  SummaryContainer &GetSummaryContainer() { return m_summary_cont; }

  FilterContainerSP GetTypeFiltersContainer() {
    return m_filter_cont.GetExactMatch();
  }

  RegexFilterContainerSP GetRegexTypeFiltersContainer() {
    return m_filter_cont.GetRegexMatch();
  }

  FilterContainer &GetFilterContainer() { return m_filter_cont; }

  FormatContainer::MapValueType
  GetFormatForType(lldb::TypeNameSpecifierImplSP type_sp);

  SummaryContainer::MapValueType
  GetSummaryForType(lldb::TypeNameSpecifierImplSP type_sp);

  FilterContainer::MapValueType
  GetFilterForType(lldb::TypeNameSpecifierImplSP type_sp);

  SynthContainer::MapValueType
  GetSyntheticForType(lldb::TypeNameSpecifierImplSP type_sp);

  lldb::TypeNameSpecifierImplSP
  GetTypeNameSpecifierForFormatAtIndex(size_t index);

  lldb::TypeNameSpecifierImplSP
  GetTypeNameSpecifierForSummaryAtIndex(size_t index);

  FormatContainer::MapValueType GetFormatAtIndex(size_t index);

  SummaryContainer::MapValueType GetSummaryAtIndex(size_t index);

  FilterContainer::MapValueType GetFilterAtIndex(size_t index);

  lldb::TypeNameSpecifierImplSP
  GetTypeNameSpecifierForFilterAtIndex(size_t index);

  SynthContainerSP GetTypeSyntheticsContainer() {
    return m_synth_cont.GetExactMatch();
  }

  RegexSynthContainerSP GetRegexTypeSyntheticsContainer() {
    return m_synth_cont.GetRegexMatch();
  }

  SynthContainer &GetSyntheticsContainer() { return m_synth_cont; }

  SynthContainer::MapValueType GetSyntheticAtIndex(size_t index);

  lldb::TypeNameSpecifierImplSP
  GetTypeNameSpecifierForSyntheticAtIndex(size_t index);

  bool IsEnabled() const { return m_enabled; }

  uint32_t GetEnabledPosition() {
    if (!m_enabled)
      return UINT32_MAX;
    else
      return m_enabled_position;
  }

  bool Get(lldb::LanguageType lang, const FormattersMatchVector &candidates,
           lldb::TypeFormatImplSP &entry);

  bool Get(lldb::LanguageType lang, const FormattersMatchVector &candidates,
           lldb::TypeSummaryImplSP &entry);

  bool Get(lldb::LanguageType lang, const FormattersMatchVector &candidates,
           lldb::SyntheticChildrenSP &entry);

  void Clear(FormatCategoryItems items = ALL_ITEM_TYPES);

  bool Delete(ConstString name, FormatCategoryItems items = ALL_ITEM_TYPES);

  uint32_t GetCount(FormatCategoryItems items = ALL_ITEM_TYPES);

  const char *GetName() { return m_name.GetCString(); }

  size_t GetNumLanguages();

  lldb::LanguageType GetLanguageAtIndex(size_t idx);

  void AddLanguage(lldb::LanguageType lang);

  std::string GetDescription();

  bool AnyMatches(ConstString type_name,
                  FormatCategoryItems items = ALL_ITEM_TYPES,
                  bool only_enabled = true,
                  const char **matching_category = nullptr,
                  FormatCategoryItems *matching_type = nullptr);

  typedef std::shared_ptr<TypeCategoryImpl> SharedPointer;

private:
  FormatContainer m_format_cont;
  SummaryContainer m_summary_cont;
  FilterContainer m_filter_cont;
  SynthContainer m_synth_cont;

  bool m_enabled;

  IFormatChangeListener *m_change_listener;

  std::recursive_mutex m_mutex;

  ConstString m_name;

  std::vector<lldb::LanguageType> m_languages;

  uint32_t m_enabled_position;

  void Enable(bool value, uint32_t position);

  void Disable() { Enable(false, UINT32_MAX); }

  bool IsApplicable(lldb::LanguageType lang);

  uint32_t GetLastEnabledPosition() { return m_enabled_position; }

  void SetEnabledPosition(uint32_t p) { m_enabled_position = p; }

  friend class FormatManager;
  friend class LanguageCategory;
  friend class TypeCategoryMap;

  friend class FormattersContainer<TypeFormatImpl>;

  friend class FormattersContainer<TypeSummaryImpl>;

  friend class FormattersContainer<TypeFilterImpl>;

  friend class FormattersContainer<ScriptedSyntheticChildren>;
};

} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_TYPECATEGORY_H

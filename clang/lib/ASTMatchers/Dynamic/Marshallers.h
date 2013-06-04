//===--- Marshallers.h - Generic matcher function marshallers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Functions templates and classes to wrap matcher construct functions.
///
/// A collection of template function and classes that provide a generic
/// marshalling layer on top of matcher construct functions.
/// These are used by the registry to export all marshaller constructors with
/// the same generic interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H

#include <list>
#include <string>
#include <vector>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/Dynamic/Diagnostics.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/type_traits.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

namespace internal {

/// \brief Helper template class to just from argument type to the right is/get
///   functions in VariantValue.
/// Used to verify and extract the matcher arguments below.
template <class T> struct ArgTypeTraits;
template <class T> struct ArgTypeTraits<const T &> : public ArgTypeTraits<T> {
};

template <> struct ArgTypeTraits<std::string> {
  static bool is(const VariantValue &Value) { return Value.isString(); }
  static const std::string &get(const VariantValue &Value) {
    return Value.getString();
  }
};

template <class T> struct ArgTypeTraits<ast_matchers::internal::Matcher<T> > {
  static bool is(const VariantValue &Value) { return Value.isMatcher(); }
  static ast_matchers::internal::Matcher<T> get(const VariantValue &Value) {
    return Value.getTypedMatcher<T>();
  }
};

template <> struct ArgTypeTraits<unsigned> {
  static bool is(const VariantValue &Value) { return Value.isUnsigned(); }
  static unsigned get(const VariantValue &Value) {
    return Value.getUnsigned();
  }
};

/// \brief Generic MatcherCreate interface.
///
/// Provides a \c run() method that constructs the matcher from the provided
/// arguments.
class MatcherCreateCallback {
public:
  virtual ~MatcherCreateCallback() {}
  virtual DynTypedMatcher *run(const SourceRange &NameRange,
                               ArrayRef<ParserValue> Args,
                               Diagnostics *Error) const = 0;
};

/// \brief Simple callback implementation. Marshaller and function are provided.
template <typename MarshallerType, typename FuncType>
class FixedArgCountMatcherCreateCallback : public MatcherCreateCallback {
public:
  /// \param Marshaller Function to unpack the arguments and call \c Func
  /// \param Func Matcher construct function. This is the function that
  ///   compile-time matcher expressions would use to create the matcher.
  FixedArgCountMatcherCreateCallback(MarshallerType Marshaller, FuncType Func,
                                     StringRef MatcherName)
      : Marshaller(Marshaller), Func(Func), MatcherName(MatcherName.str()) {}

  DynTypedMatcher *run(const SourceRange &NameRange,
                       ArrayRef<ParserValue> Args, Diagnostics *Error) const {
    return Marshaller(Func, MatcherName, NameRange, Args, Error);
  }

private:
  const MarshallerType Marshaller;
  const FuncType Func;
  const std::string MatcherName;
};

/// \brief Helper function to do template argument deduction.
template <typename MarshallerType, typename FuncType>
MatcherCreateCallback *
createMarshallerCallback(MarshallerType Marshaller, FuncType Func,
                         StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<MarshallerType, FuncType>(
      Marshaller, Func, MatcherName);
}

/// \brief Helper macros to check the arguments on all marshaller functions.
#define CHECK_ARG_COUNT(count)                                                 \
  if (Args.size() != count) {                                                  \
    Error->pushErrorFrame(NameRange, Error->ET_RegistryWrongArgCount)          \
        << count << Args.size();                                               \
    return NULL;                                                               \
  }

#define CHECK_ARG_TYPE(index, type)                                            \
  if (!ArgTypeTraits<type>::is(Args[index].Value)) {                           \
    Error->pushErrorFrame(Args[index].Range, Error->ET_RegistryWrongArgType)   \
        << MatcherName << (index + 1);                                         \
    return NULL;                                                               \
  }

/// \brief 0-arg marshaller function.
template <typename ReturnType>
DynTypedMatcher *matcherMarshall0(ReturnType (*Func)(), StringRef MatcherName,
                                  const SourceRange &NameRange,
                                  ArrayRef<ParserValue> Args,
                                  Diagnostics *Error) {
  CHECK_ARG_COUNT(0);
  return Func().clone();
}

/// \brief 1-arg marshaller function.
template <typename ReturnType, typename ArgType1>
DynTypedMatcher *matcherMarshall1(ReturnType (*Func)(ArgType1),
                                  StringRef MatcherName,
                                  const SourceRange &NameRange,
                                  ArrayRef<ParserValue> Args,
                                  Diagnostics *Error) {
  CHECK_ARG_COUNT(1);
  CHECK_ARG_TYPE(0, ArgType1);
  return Func(ArgTypeTraits<ArgType1>::get(Args[0].Value)).clone();
}

/// \brief 2-arg marshaller function.
template <typename ReturnType, typename ArgType1, typename ArgType2>
DynTypedMatcher *matcherMarshall2(ReturnType (*Func)(ArgType1, ArgType2),
                                  StringRef MatcherName,
                                  const SourceRange &NameRange,
                                  ArrayRef<ParserValue> Args,
                                  Diagnostics *Error) {
  CHECK_ARG_COUNT(2);
  CHECK_ARG_TYPE(0, ArgType1);
  CHECK_ARG_TYPE(1, ArgType2);
  return Func(ArgTypeTraits<ArgType1>::get(Args[0].Value),
              ArgTypeTraits<ArgType2>::get(Args[1].Value)).clone();
}

/// \brief Variadic marshaller function.
template <typename BaseType, typename DerivedType>
class VariadicMatcherCreateCallback : public MatcherCreateCallback {
public:
  explicit VariadicMatcherCreateCallback(StringRef MatcherName)
      : MatcherName(MatcherName.str()) {}

  typedef ast_matchers::internal::Matcher<DerivedType> DerivedMatcherType;

  DynTypedMatcher *run(const SourceRange &NameRange, ArrayRef<ParserValue> Args,
                       Diagnostics *Error) const {
    std::list<DerivedMatcherType> References;
    std::vector<const DerivedMatcherType *> InnerArgs(Args.size());
    for (size_t i = 0, e = Args.size(); i != e; ++i) {
      CHECK_ARG_TYPE(i, DerivedMatcherType);
      References.push_back(
          ArgTypeTraits<DerivedMatcherType>::get(Args[i].Value));
      InnerArgs[i] = &References.back();
    }
    return ast_matchers::internal::makeDynCastAllOfComposite<BaseType>(
        ArrayRef<const DerivedMatcherType *>(InnerArgs)).clone();
  }

private:
  const std::string MatcherName;
};

#undef CHECK_ARG_COUNT
#undef CHECK_ARG_TYPE

/// Helper functions to select the appropriate marshaller functions.
/// They detects the number of arguments, arguments types and return type.

/// \brief 0-arg overload
template <typename ReturnType>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall0<ReturnType>, Func,
                                  MatcherName);
}

/// \brief 1-arg overload
template <typename ReturnType, typename ArgType1>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                                               StringRef MatcherName) {
  return createMarshallerCallback(matcherMarshall1<ReturnType, ArgType1>, Func,
                                  MatcherName);
}

/// \brief 2-arg overload
template <typename ReturnType, typename ArgType1, typename ArgType2>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1,
                                                                  ArgType2),
                                               StringRef MatcherName) {
  return createMarshallerCallback(
      matcherMarshall2<ReturnType, ArgType1, ArgType2>, Func, MatcherName);
}

/// \brief Variadic overload.
template <typename MatcherType>
MatcherCreateCallback *makeMatcherAutoMarshall(
    ast_matchers::internal::VariadicAllOfMatcher<MatcherType> Func,
    StringRef MatcherName) {
  return new VariadicMatcherCreateCallback<MatcherType, MatcherType>(
      MatcherName);
}

template <typename BaseType, typename MatcherType>
MatcherCreateCallback *
makeMatcherAutoMarshall(ast_matchers::internal::VariadicDynCastAllOfMatcher<
                            BaseType, MatcherType> Func,
                        StringRef MatcherName) {
  return new VariadicMatcherCreateCallback<BaseType, MatcherType>(MatcherName);
}

}  // namespace internal
}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H

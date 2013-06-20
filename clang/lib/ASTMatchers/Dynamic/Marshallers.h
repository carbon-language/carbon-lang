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

#include <string>

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
  static StringRef asString() { return "String"; }
  static bool is(const VariantValue &Value) { return Value.isString(); }
  static const std::string &get(const VariantValue &Value) {
    return Value.getString();
  }
};

template <class T> struct ArgTypeTraits<ast_matchers::internal::Matcher<T> > {
  static std::string asString() {
    return (Twine("Matcher<") +
            ast_type_traits::ASTNodeKind::getFromNodeKind<T>().asStringRef() +
            ">").str();
  }
  static bool is(const VariantValue &Value) {
    return Value.hasTypedMatcher<T>();
  }
  static ast_matchers::internal::Matcher<T> get(const VariantValue &Value) {
    return Value.getTypedMatcher<T>();
  }
};

template <> struct ArgTypeTraits<unsigned> {
  static std::string asString() { return "Unsigned"; }
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
///
/// This class wraps a function of arbitrary signature and a marshaller
/// function into a MatcherCreateCallback.
/// The marshaller is in charge of taking the VariantValue arguments, checking
/// their types, unpacking them and calling the underlying function.
template <typename FuncType>
class FixedArgCountMatcherCreateCallback : public MatcherCreateCallback {
public:
  /// FIXME: Use void(*)() as FuncType on this interface to remove the template
  /// argument of this class. The marshaller can cast the function pointer back
  /// to the original type.
  typedef DynTypedMatcher *(*MarshallerType)(FuncType, StringRef,
                                             const SourceRange &,
                                             ArrayRef<ParserValue>,
                                             Diagnostics *);

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

/// \brief Simple callback implementation. Free function is wrapped.
///
/// This class simply wraps a free function with the right signature to export
/// it as a MatcherCreateCallback.
/// This allows us to have one implementation of the interface for as many free
/// functions as we want, reducing the number of symbols and size of the
/// object file.
class FreeFuncMatcherCreateCallback : public MatcherCreateCallback {
public:
  typedef DynTypedMatcher *(*RunFunc)(StringRef MatcherName,
                                      const SourceRange &NameRange,
                                      ArrayRef<ParserValue> Args,
                                      Diagnostics *Error);

  FreeFuncMatcherCreateCallback(RunFunc Func, StringRef MatcherName)
      : Func(Func), MatcherName(MatcherName.str()) {}

  DynTypedMatcher *run(const SourceRange &NameRange, ArrayRef<ParserValue> Args,
                       Diagnostics *Error) const {
    return Func(MatcherName, NameRange, Args, Error);
  }

private:
  const RunFunc Func;
  const std::string MatcherName;
};

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
        << (index + 1) << ArgTypeTraits<type>::asString()                      \
        << Args[index].Value.getTypeAsString();                                \
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

#undef CHECK_ARG_COUNT
#undef CHECK_ARG_TYPE

/// \brief Variadic marshaller function.
template <typename BaseType, typename DerivedType>
DynTypedMatcher *VariadicMatcherCreateCallback(StringRef MatcherName,
                                               const SourceRange &NameRange,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) {
  typedef ast_matchers::internal::Matcher<DerivedType> DerivedMatcherType;
  DerivedMatcherType **InnerArgs = new DerivedMatcherType *[Args.size()]();

  bool HasError = false;
  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    typedef ArgTypeTraits<DerivedMatcherType> DerivedTraits;
    const ParserValue &Arg = Args[i];
    const VariantValue &Value = Arg.Value;
    if (!DerivedTraits::is(Value)) {
      Error->pushErrorFrame(Arg.Range, Error->ET_RegistryWrongArgType)
          << (i + 1) << DerivedTraits::asString() << Value.getTypeAsString();
      HasError = true;
      break;
    }
    InnerArgs[i] = new DerivedMatcherType(DerivedTraits::get(Value));
  }

  DynTypedMatcher *Out = NULL;
  if (!HasError) {
    Out = ast_matchers::internal::makeDynCastAllOfComposite<BaseType>(
        ArrayRef<const DerivedMatcherType *>(InnerArgs, Args.size())).clone();
  }

  for (size_t i = 0, e = Args.size(); i != e; ++i) {
    delete InnerArgs[i];
  }
  delete[] InnerArgs;
  return Out;
}

/// Helper functions to select the appropriate marshaller functions.
/// They detect the number of arguments, arguments types and return type.

/// \brief 0-arg overload
template <typename ReturnType>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(),
                                               StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<ReturnType (*)()>(
      matcherMarshall0, Func, MatcherName);
}

/// \brief 1-arg overload
template <typename ReturnType, typename ArgType1>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1),
                                               StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<ReturnType (*)(ArgType1)>(
      matcherMarshall1, Func, MatcherName);
}

/// \brief 2-arg overload
template <typename ReturnType, typename ArgType1, typename ArgType2>
MatcherCreateCallback *makeMatcherAutoMarshall(ReturnType (*Func)(ArgType1,
                                                                  ArgType2),
                                               StringRef MatcherName) {
  return new FixedArgCountMatcherCreateCallback<
      ReturnType (*)(ArgType1, ArgType2)>(matcherMarshall2, Func, MatcherName);
}

/// \brief Variadic overload.
template <typename MatcherType>
MatcherCreateCallback *makeMatcherAutoMarshall(
    ast_matchers::internal::VariadicAllOfMatcher<MatcherType> Func,
    StringRef MatcherName) {
  return new FreeFuncMatcherCreateCallback(
      &VariadicMatcherCreateCallback<MatcherType, MatcherType>, MatcherName);
}

template <typename BaseType, typename MatcherType>
MatcherCreateCallback *
makeMatcherAutoMarshall(ast_matchers::internal::VariadicDynCastAllOfMatcher<
                            BaseType, MatcherType> Func,
                        StringRef MatcherName) {
  return new FreeFuncMatcherCreateCallback(
      &VariadicMatcherCreateCallback<BaseType, MatcherType>, MatcherName);
}

}  // namespace internal
}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_MARSHALLERS_H

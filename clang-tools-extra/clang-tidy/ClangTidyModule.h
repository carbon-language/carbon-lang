//===--- ClangTidyModule.h - clang-tidy -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_H

#include "ClangTidy.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

namespace clang {
namespace tidy {

/// \brief A factory, that can instantiate a specific clang-tidy check for
/// processing a translation unit.
///
/// In order to register your check with the \c ClangTidyModule, create a
/// subclass of \c CheckFactoryBase and implement \c createCheck(). Then, use
/// this subclass in \c ClangTidyModule::addCheckFactories().
class CheckFactoryBase {
public:
  virtual ~CheckFactoryBase() {}
  virtual ClangTidyCheck *createCheck() = 0;
};

/// \brief A subclass of \c CheckFactoryBase that should be used for all
/// \c ClangTidyChecks that don't require constructor parameters.
///
/// For example, if have a clang-tidy check like:
/// \code
/// class MyTidyCheck : public ClangTidyCheck {
///   void registerMatchers(ast_matchers::MatchFinder *Finder) override {
///     ..
///   }
/// };
/// \endcode
/// you can register it with:
/// \code
/// class MyModule : public ClangTidyModule {
///   void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
///     CheckFactories.addCheckFactory(
///         "myproject-my-check", new ClangTidyCheckFactory<MyTidyCheck>());
///   }
/// };
/// \endcode
template <typename T> class ClangTidyCheckFactory : public CheckFactoryBase {
public:
  ClangTidyCheck *createCheck() override { return new T; }
};

class ClangTidyCheckFactories;

/// \brief A clang-tidy module groups a number of \c ClangTidyChecks and gives
/// them a prefixed name.
class ClangTidyModule {
public:
  virtual ~ClangTidyModule() {}

  /// \brief Implement this function in order to register all \c CheckFactories
  /// belonging to this module.
  virtual void addCheckFactories(ClangTidyCheckFactories &CheckFactories) = 0;
};

/// \brief A collection of \c ClangTidyCheckFactory instances.
///
/// All clang-tidy modules register their check factories with an instance of
/// this object.
class ClangTidyCheckFactories {
public:
  ClangTidyCheckFactories() {}
  ~ClangTidyCheckFactories();

  /// \brief Register \p Factory with the name \p Name.
  ///
  /// The \c ClangTidyCheckFactories object takes ownership of the \p Factory.
  void addCheckFactory(StringRef Name, CheckFactoryBase *Factory);

  /// \brief Create instances of all checks matching \p CheckRegexString and
  /// store them in \p Checks.
  ///
  /// The caller takes ownership of the return \c ClangTidyChecks.
  void createChecks(ChecksFilter &Filter,
                    SmallVectorImpl<ClangTidyCheck *> &Checks);

  typedef std::map<std::string, CheckFactoryBase *> FactoryMap;
  FactoryMap::const_iterator begin() const { return Factories.begin(); }
  FactoryMap::const_iterator end() const { return Factories.end(); }
  bool empty() const { return Factories.empty(); }

private:
  FactoryMap Factories;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_MODULE_H

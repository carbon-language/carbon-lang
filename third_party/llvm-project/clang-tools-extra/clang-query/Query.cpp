//===---- Query.cpp - clang-query query -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "clang/AST/ASTDumper.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Tooling/NodeIntrospection.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;

namespace clang {
namespace query {

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << ErrStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER          "
        "Match the loaded ASTs against the given matcher.\n"
        "  let NAME MATCHER, l NAME MATCHER  "
        "Give a matcher expression a name, to be used later\n"
        "                                    "
        "as part of other expressions.\n"
        "  set bind-root (true|false)        "
        "Set whether to bind the root matcher to \"root\".\n"
        "  set print-matcher (true|false)    "
        "Set whether to print the current matcher,\n"
        "  set traversal <kind>              "
        "Set traversal kind of clang-query session. Available kinds are:\n"
        "    AsIs                            "
        "Print and match the AST as clang sees it.  This mode is the "
        "default.\n"
        "    IgnoreUnlessSpelledInSource     "
        "Omit AST nodes unless spelled in the source.\n"
        "  set output <feature>              "
        "Set whether to output only <feature> content.\n"
        "  enable output <feature>           "
        "Enable <feature> content non-exclusively.\n"
        "  disable output <feature>          "
        "Disable <feature> content non-exclusively.\n"
        "  quit, q                           "
        "Terminates the query session.\n\n"
        "Several commands accept a <feature> parameter. The available features "
        "are:\n\n"
        "  print                             "
        "Pretty-print bound nodes.\n"
        "  diag                              "
        "Diagnostic location for bound nodes.\n"
        "  detailed-ast                      "
        "Detailed AST output for bound nodes.\n"
        "  srcloc                            "
        "Source locations and ranges for bound nodes.\n"
        "  dump                              "
        "Detailed AST output for bound nodes (alias of detailed-ast).\n\n";
  return true;
}

bool QuitQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  QS.Terminate = true;
  return true;
}

namespace {

struct CollectBoundNodes : MatchFinder::MatchCallback {
  std::vector<BoundNodes> &Bindings;
  CollectBoundNodes(std::vector<BoundNodes> &Bindings) : Bindings(Bindings) {}
  void run(const MatchFinder::MatchResult &Result) override {
    Bindings.push_back(Result.Nodes);
  }
};

void dumpLocations(llvm::raw_ostream &OS, DynTypedNode Node, ASTContext &Ctx,
                   const DiagnosticsEngine &Diags, SourceManager const &SM) {
  auto Locs = clang::tooling::NodeIntrospection::GetLocations(Node);

  auto PrintLocations = [](llvm::raw_ostream &OS, auto Iter, auto End) {
    auto CommonEntry = Iter->first;
    auto Scout = Iter;
    SmallVector<std::string> LocationStrings;
    while (Scout->first == CommonEntry) {
      LocationStrings.push_back(
          tooling::LocationCallFormatterCpp::format(*Iter->second));
      if (Scout == End)
        break;
      ++Scout;
      if (Scout->first == CommonEntry)
        ++Iter;
    }
    llvm::sort(LocationStrings);
    for (auto &LS : LocationStrings) {
      OS << " * \"" << LS << "\"\n";
    }
    return Iter;
  };

  TextDiagnostic TD(OS, Ctx.getLangOpts(), &Diags.getDiagnosticOptions());

  for (auto Iter = Locs.LocationAccessors.begin();
       Iter != Locs.LocationAccessors.end(); ++Iter) {
    if (!Iter->first.isValid())
      continue;

    TD.emitDiagnostic(FullSourceLoc(Iter->first, SM), DiagnosticsEngine::Note,
                      "source locations here", None, None);

    Iter = PrintLocations(OS, Iter, Locs.LocationAccessors.end());
    OS << '\n';
  }

  for (auto Iter = Locs.RangeAccessors.begin();
       Iter != Locs.RangeAccessors.end(); ++Iter) {

    if (!Iter->first.getBegin().isValid())
      continue;

    if (SM.getPresumedLineNumber(Iter->first.getBegin()) !=
        SM.getPresumedLineNumber(Iter->first.getEnd()))
      continue;

    TD.emitDiagnostic(FullSourceLoc(Iter->first.getBegin(), SM),
                      DiagnosticsEngine::Note,
                      "source ranges here " + Iter->first.printToString(SM),
                      CharSourceRange::getTokenRange(Iter->first), None);

    Iter = PrintLocations(OS, Iter, Locs.RangeAccessors.end());
  }
  for (auto Iter = Locs.RangeAccessors.begin();
       Iter != Locs.RangeAccessors.end(); ++Iter) {

    if (!Iter->first.getBegin().isValid())
      continue;

    if (SM.getPresumedLineNumber(Iter->first.getBegin()) ==
        SM.getPresumedLineNumber(Iter->first.getEnd()))
      continue;

    TD.emitDiagnostic(
        FullSourceLoc(Iter->first.getBegin(), SM), DiagnosticsEngine::Note,
        "source range " + Iter->first.printToString(SM) + " starting here...",
        CharSourceRange::getTokenRange(Iter->first), None);

    auto ColNum = SM.getPresumedColumnNumber(Iter->first.getEnd());
    auto LastLineLoc = Iter->first.getEnd().getLocWithOffset(-(ColNum - 1));

    TD.emitDiagnostic(FullSourceLoc(Iter->first.getEnd(), SM),
                      DiagnosticsEngine::Note, "... ending here",
                      CharSourceRange::getTokenRange(
                          SourceRange(LastLineLoc, Iter->first.getEnd())),
                      None);

    Iter = PrintLocations(OS, Iter, Locs.RangeAccessors.end());
  }
  OS << "\n";
}

} // namespace

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  for (auto &AST : QS.ASTs) {
    MatchFinder Finder;
    std::vector<BoundNodes> Matches;
    DynTypedMatcher MaybeBoundMatcher = Matcher;
    if (QS.BindRoot) {
      llvm::Optional<DynTypedMatcher> M = Matcher.tryBind("root");
      if (M)
        MaybeBoundMatcher = *M;
    }
    CollectBoundNodes Collect(Matches);
    if (!Finder.addDynamicMatcher(MaybeBoundMatcher, &Collect)) {
      OS << "Not a valid top-level matcher.\n";
      return false;
    }

    auto &Ctx = AST->getASTContext();
    const auto &SM = Ctx.getSourceManager();
    Ctx.getParentMapContext().setTraversalKind(QS.TK);
    Finder.matchAST(Ctx);

    if (QS.PrintMatcher) {
      SmallVector<StringRef, 4> Lines;
      Source.split(Lines, "\n");
      auto FirstLine = Lines[0];
      Lines.erase(Lines.begin(), Lines.begin() + 1);
      while (!Lines.empty() && Lines.back().empty()) {
        Lines.resize(Lines.size() - 1);
      }
      unsigned MaxLength = FirstLine.size();
      std::string PrefixText = "Matcher: ";
      OS << "\n  " << PrefixText << FirstLine;

      for (auto Line : Lines) {
        OS << "\n" << std::string(PrefixText.size() + 2, ' ') << Line;
        MaxLength = std::max<int>(MaxLength, Line.rtrim().size());
      }

      OS << "\n"
         << "  " << std::string(PrefixText.size() + MaxLength, '=') << "\n\n";
    }

    for (auto MI = Matches.begin(), ME = Matches.end(); MI != ME; ++MI) {
      OS << "\nMatch #" << ++MatchCount << ":\n\n";

      for (auto BI = MI->getMap().begin(), BE = MI->getMap().end(); BI != BE;
           ++BI) {
        if (QS.DiagOutput) {
          clang::SourceRange R = BI->second.getSourceRange();
          if (R.isValid()) {
            TextDiagnostic TD(OS, AST->getASTContext().getLangOpts(),
                              &AST->getDiagnostics().getDiagnosticOptions());
            TD.emitDiagnostic(
                FullSourceLoc(R.getBegin(), AST->getSourceManager()),
                DiagnosticsEngine::Note, "\"" + BI->first + "\" binds here",
                CharSourceRange::getTokenRange(R), None);
          }
        }
        if (QS.PrintOutput) {
          OS << "Binding for \"" << BI->first << "\":\n";
          BI->second.print(OS, AST->getASTContext().getPrintingPolicy());
          OS << "\n";
        }
        if (QS.DetailedASTOutput) {
          OS << "Binding for \"" << BI->first << "\":\n";
          const ASTContext &Ctx = AST->getASTContext();
          ASTDumper Dumper(OS, Ctx, AST->getDiagnostics().getShowColors());
          Dumper.SetTraversalKind(QS.TK);
          Dumper.Visit(BI->second);
          OS << "\n";
        }
        if (QS.SrcLocOutput) {
          OS << "\n  \"" << BI->first << "\" Source locations\n";
          OS << "  " << std::string(19 + BI->first.size(), '-') << '\n';

          dumpLocations(OS, BI->second, Ctx, AST->getDiagnostics(), SM);
          OS << "\n";
        }
      }

      if (MI->getMap().empty())
        OS << "No bindings.\n";
    }
  }

  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

bool LetQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  if (Value) {
    QS.NamedValues[Name] = Value;
  } else {
    QS.NamedValues.erase(Name);
  }
  return true;
}

#ifndef _MSC_VER
const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;
#endif

} // namespace query
} // namespace clang

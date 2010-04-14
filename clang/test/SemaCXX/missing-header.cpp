// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "not exist" // expected-error{{'not exist' file not found}}

class AnalysisContext {};
static ControlFlowKind CheckFallThrough(AnalysisContext &AC) {
  if (const AsmStmt *AS = dyn_cast<AsmStmt>(S)) {}
  bool NoReturnEdge = false;
}

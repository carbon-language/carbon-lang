//===-- Solution.h ------- PBQP Solution ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PBQP Solution class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_SOLUTION_H
#define LLVM_CODEGEN_PBQP_SOLUTION_H

#include "Math.h"
#include "Graph.h"

#include <map>

namespace PBQP {

  /// \brief Represents a solution to a PBQP problem.
  ///
  /// To get the selection for each node in the problem use the getSelection method.
  class Solution {
  private:
    typedef std::map<Graph::NodeItr, unsigned, NodeItrComparator> SelectionsMap;
    SelectionsMap selections;

  public:

    /// \brief Number of nodes for which selections have been made.
    /// @return Number of nodes for which selections have been made.
    unsigned numNodes() const { return selections.size(); }

    /// \brief Set the selection for a given node.
    /// @param nItr Node iterator.
    /// @param selection Selection for nItr.
    void setSelection(Graph::NodeItr nItr, unsigned selection) {
      selections[nItr] = selection;
    }

    /// \brief Get a node's selection.
    /// @param nItr Node iterator.
    /// @return The selection for nItr;
    unsigned getSelection(Graph::NodeItr nItr) const {
      SelectionsMap::const_iterator sItr = selections.find(nItr);
      assert(sItr != selections.end() && "No selection for node.");
      return sItr->second;
    }

  };

}

#endif // LLVM_CODEGEN_PBQP_SOLUTION_H
